import argparse
import pyhocon
from sklearn.utils import shuffle
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from itertools import combinations

from models import SpanEmbedder, SpanScorer, SimplePairWiseClassifier
from evaluator import Evaluation
from spans import TopicSpans
from model_utils import *
from utils import *



def train_pairwise_classifier(config, pairwise_model, span_repr, span_scorer, span_embeddings,
                                    first, second, labels, batch_size, criterion, optimizer):
    accumulate_loss = 0
    start_end_embeddings, continuous_embeddings, width = span_embeddings
    device = start_end_embeddings.device
    labels = labels.to(device)
    # width = width.to(device)

    idx = shuffle(list(range(len(first))))
    for i in range(0, len(first), batch_size):
        indices = idx[i:i+batch_size]
        batch_first, batch_second = first[indices], second[indices]
        batch_labels = labels[indices].to(torch.float)
        optimizer.zero_grad()
        g1 = span_repr(start_end_embeddings[batch_first],
                                [continuous_embeddings[k] for k in batch_first], width[batch_first])
        g2 = span_repr(start_end_embeddings[batch_second],
                                [continuous_embeddings[k] for k in batch_second], width[batch_second])
        scores = pairwise_model(g1, g2)

        if config['training_method'] in ('continue', 'e2e') and not config['use_gold_mentions']:
            g1_score = span_scorer(g1)
            g2_score = span_scorer(g2)
            scores += g1_score + g2_score

        loss = criterion(scores.squeeze(1), batch_labels)
        accumulate_loss += loss.item()
        loss.backward()
        optimizer.step()

        torch.cuda.empty_cache()

    return accumulate_loss


def get_all_candidate_spans(config, bert_model, span_repr, span_scorer, data, topic_num):
    docs_embeddings, docs_length = pad_and_read_bert(data.topics_bert_tokens[topic_num], bert_model)
    topic_spans = TopicSpans(config, data, topic_num, docs_embeddings, docs_length, is_training=True)

    topic_spans.set_span_labels()

    ## Pruning the spans according to gold mentions or spans with highiest scores
    if config['use_gold_mentions']:
        span_indices = torch.nonzero(topic_spans.labels).squeeze(1)
    else:
        with torch.no_grad():
            span_emb = span_repr(topic_spans.start_end_embeddings,
                                 topic_spans.continuous_embeddings,
                                 topic_spans.width)
            span_scores = span_scorer(span_emb)

        if config.exact:
            span_indices = torch.where(span_scores > 0)[0]
        else:
            k = int(config['top_k'] * topic_spans.num_tokens)
            _, span_indices = torch.topk(span_scores.squeeze(1), k, sorted=False)

    span_indices = span_indices.cpu()
    topic_spans.prune_spans(span_indices)
    torch.cuda.empty_cache()

    return topic_spans


def get_pairwise_labels(labels, is_training):
    first, second = zip(*list(combinations(range(len(labels)), 2)))
    first = torch.tensor(first)
    second = torch.tensor(second)
    pairwise_labels = (labels[first] != 0) & (labels[second] != 0) & \
                      (labels[first] == labels[second])

    if is_training:
        positives = torch.nonzero(pairwise_labels == 1).squeeze()
        positive_ratio = len(positives) / len(first)
        negatives = torch.nonzero(pairwise_labels != 1).squeeze()
        rands = torch.rand(len(negatives))
        rands = (rands < positive_ratio * 20).to(torch.long)
        sampled_negatives = negatives[torch.nonzero(rands).squeeze()]
        new_first = torch.cat((first[positives], first[sampled_negatives]))
        new_second = torch.cat((second[positives], second[sampled_negatives]))
        new_labels = torch.cat((pairwise_labels[positives], pairwise_labels[sampled_negatives]))
        first, second, pairwise_labels = new_first, new_second, new_labels


    pairwise_labels = pairwise_labels.to(torch.long).to(device)

    if config['loss'] == 'hinge':
        pairwise_labels = torch.where(pairwise_labels == 1, pairwise_labels, torch.tensor(-1, device=device))
    else:
        pairwise_labels = torch.where(pairwise_labels == 1, pairwise_labels, torch.tensor(0, device=device))
    torch.cuda.empty_cache()


    return first, second, pairwise_labels



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config_pairwise.json')
    args = parser.parse_args()


    config = pyhocon.ConfigFactory.parse_file(args.config)
    fix_seed(config)
    logger = create_logger(config, create_file=True)
    logger.info(pyhocon.HOCONConverter.convert(config, "hocon"))
    create_folder(config['model_path'])

    device = torch.device('cuda:{}'.format(config.gpu_num[0])) if torch.cuda.is_available() else 'cpu'

    logger.info('Using device {}'.format(device))
    # init train and dev set
    bert_tokenizer = AutoTokenizer.from_pretrained(config['bert_model'])
    training_set = create_corpus(config, bert_tokenizer, 'train')
    dev_set = create_corpus(config, bert_tokenizer, 'dev')


    ## Model initiation
    logger.info('Init models')
    bert_model = AutoModel.from_pretrained(config['bert_model']).to(device)
    config['bert_hidden_size'] = bert_model.config.hidden_size

    span_repr = SpanEmbedder(config, device).to(device)
    span_scorer = SpanScorer(config).to(device)

    if config['training_method'] in ('pipeline', 'continue') and not config['use_gold_mentions']:
        span_repr.load_state_dict(torch.load(config['span_repr_path'], map_location=device))
        span_scorer.load_state_dict(torch.load(config['span_scorer_path'], map_location=device))

    span_repr.eval()
    span_scorer.eval()

    pairwise_model = SimplePairWiseClassifier(config).to(device)


    ## Optimizer and loss function
    models = [pairwise_model]
    if config['training_method'] in ('continue', 'e2e') and not config['use_gold_mentions']:
        models.append(span_repr)
        models.append(span_scorer)
    optimizer = get_optimizer(config, models)
    criterion = get_loss_function(config)


    logger.info('Number of parameters of mention extractor: {}'.format(
        count_parameters(span_repr) + count_parameters(span_scorer)))
    logger.info('Number of parameters of the pairwise classifier: {}'.format(
        count_parameters(pairwise_model)))

    logger.info('Number of topics: {}'.format(len(training_set.topic_list)))
    f1 = []
    for epoch in range(config['epochs']):
        logger.info('Epoch: {}'.format(epoch))

        pairwise_model.train()
        if config['training_method'] in ('continue', 'e2e') and not config['use_gold_mentions']:
            span_repr.train()
            span_scorer.train()

        accumulate_loss = 0
        list_of_topics = shuffle(list(range(len(training_set.topic_list))))
        total_number_of_pairs = 0
        for topic_num in tqdm(list_of_topics):
            topic = training_set.topic_list[topic_num]
            topic_spans = get_all_candidate_spans(config, bert_model, span_repr, span_scorer, training_set, topic_num)
            first, second, pairwise_labels = get_pairwise_labels(topic_spans.labels, is_training=config['neg_samp'])
            span_embeddings = topic_spans.start_end_embeddings, topic_spans.continuous_embeddings, topic_spans.width
            loss = train_pairwise_classifier(config, pairwise_model, span_repr, span_scorer, span_embeddings, first,
                                                   second, pairwise_labels, config['batch_size'], criterion, optimizer)
            torch.cuda.empty_cache()
            accumulate_loss += loss
            total_number_of_pairs += len(first)

        logger.info('Number of training pairs: {}'.format(total_number_of_pairs))
        logger.info('Accumulate loss: {}'.format(accumulate_loss))


        logger.info('Evaluate on the dev set')

        span_repr.eval()
        span_scorer.eval()
        pairwise_model.eval()

        all_scores, all_labels = [], []

        for topic_num, topic in enumerate(tqdm(dev_set.topic_list)):
            topic_spans = get_all_candidate_spans(config, bert_model, span_repr, span_scorer, dev_set, topic_num)
            # logger.info('Topic: {}'.format(topic_num))
            # logger.info('Num of labels: {}'.format(len(topic_spans.labels)))
            first, second, pairwise_labels = get_pairwise_labels(topic_spans.labels, is_training=False)

            span_embeddings = topic_spans.start_end_embeddings, topic_spans.continuous_embeddings, \
                              topic_spans.width
            topic_spans.width = topic_spans.width.to(device)
            with torch.no_grad():
                for i in range(0, len(first), 10000):
                    end_max = i + 10000
                    first_idx, second_idx = first[i:end_max], second[i:end_max]
                    batch_labels = pairwise_labels[i:end_max]
                    g1 = span_repr(topic_spans.start_end_embeddings[first_idx],
                                   [topic_spans.continuous_embeddings[k] for k in first_idx],
                                   topic_spans.width[first_idx])
                    g2 = span_repr(topic_spans.start_end_embeddings[second_idx],
                                   [topic_spans.continuous_embeddings[k] for k in second_idx],
                                   topic_spans.width[second_idx])
                    scores = pairwise_model(g1, g2)

                    if config['training_method'] in ('continue', 'e2e') and not config['use_gold_mentions']:
                        g1_score = span_scorer(g1)
                        g2_score = span_scorer(g2)
                        scores += g1_score + g2_score

                    all_scores.extend(scores.squeeze(1))
                    all_labels.extend(batch_labels.to(torch.int))

        all_labels = torch.stack(all_labels)
        all_scores = torch.stack(all_scores)


        strict_preds = (all_scores > 0).to(torch.int)
        eval = Evaluation(strict_preds, all_labels.to(device))
        logger.info('Number of predictions: {}/{}'.format(strict_preds.sum(), len(strict_preds)))
        logger.info('Number of positive pairs: {}/{}'.format(len(torch.nonzero(all_labels == 1)),
                                                             len(all_labels)))
        logger.info('Strict - Recall: {}, Precision: {}, F1: {}'.format(eval.get_recall(),
                                                                        eval.get_precision(), eval.get_f1()))
        f1.append(eval.get_f1())

        torch.save(span_repr.state_dict(), os.path.join(config['model_path'], 'span_repr_{}'.format(epoch)))
        torch.save(span_scorer.state_dict(), os.path.join(config['model_path'], 'span_scorer_{}'.format(epoch)))
        torch.save(pairwise_model.state_dict(), os.path.join(config['model_path'], 'pairwise_scorer_{}'.format(epoch)))