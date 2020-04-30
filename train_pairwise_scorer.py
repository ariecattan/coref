import argparse
import json
import pyhocon
from sklearn.utils import shuffle
from models import SimplePairWiseClassifier, SpanEmbedder, SpanScorer
from transformers import RobertaTokenizer, RobertaModel
from evaluator import Evaluation
from tqdm import tqdm

from model_utils import *
from utils import *




parser = argparse.ArgumentParser()
parser.add_argument('--data_folder', type=str, default='data/ecb/mentions')
parser.add_argument('--config_model_file', type=str, default='configs/config_pairwise.json')
args = parser.parse_args()




def train_pairwise_classifier(config, pairwise_model, span_repr, span_scorer, span_embeddings,
                                    first, second, labels, batch_size, criterion, optimizer):
    accumulate_loss = 0
    start_end_embeddings, continuous_embeddings, width = span_embeddings
    device = start_end_embeddings.device
    labels = labels.to(device)
    width = width.to(device)

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

        if config['training_method'] in ('fine_tune', 'e2e'):
            g1_score = span_scorer(g1)
            g2_score = span_scorer(g2)
            scores += g1_score + g2_score

        loss = criterion(scores.squeeze(1), batch_labels)
        accumulate_loss += loss.item()
        loss.backward()
        optimizer.step()

    return accumulate_loss




def get_all_pairs_from_topic(config, bert_model, span_repr, span_scorer,
                             list_of_docs, docs_original_tokens, bert_tokens, docs_bert_start_end,
                             event_labels, entity_labels):
    docs_embeddings, docs_length = pad_and_read_bert(bert_tokens, bert_model)
    span_meta_data, span_embeddings, num_of_tokens = get_all_candidate_from_topic(
        config, list_of_docs, docs_original_tokens, docs_bert_start_end, docs_embeddings, docs_length)
    doc_id, sentence_id, start, end = span_meta_data
    topic_start_end_embeddings, topic_continuous_embeddings, topic_width = span_embeddings
    device = topic_start_end_embeddings.device
    topic_width = topic_width.to(device)
    pairwise_event_labels, pairwise_entity_labels = get_candidate_labels(doc_id, start, end, event_labels,
                                                                   entity_labels)
    labels = pairwise_event_labels if config['is_event'] else pairwise_entity_labels

    if config['use_gold_mentions']:
        span_indices = labels.nonzero().squeeze()

    else:
        k = int(config['top_k'] * num_of_tokens)
        with torch.no_grad():
            span_emb = span_repr(topic_start_end_embeddings, topic_continuous_embeddings, topic_width)
            span_scores = span_scorer(span_emb)
        _, span_indices = torch.topk(span_scores.squeeze(1), k, sorted=False)


    torch.cuda.empty_cache()
    labels = labels[span_indices]
    topic_start_end_embeddings = topic_start_end_embeddings[span_indices]
    topic_continuous_embeddings = [topic_continuous_embeddings[i] for i in span_indices]
    topic_width = topic_width[span_indices]
    span_embeddings = topic_start_end_embeddings, topic_continuous_embeddings, topic_width

    first, second = zip(*list(combinations(range(len(span_indices)), 2)))
    first = torch.tensor(first)
    second = torch.tensor(second)
    pairwise_labels = (labels[first] != 0) & (labels[second] != 0) & \
                      (labels[first] == labels[second])
    pairwise_labels = pairwise_labels.to(torch.long).to(device)

    if config['loss'] == 'hinge':
        pairwise_labels = torch.where(pairwise_labels == 1, pairwise_labels, torch.tensor(-1, device=device))
    else:
        pairwise_labels = torch.where(pairwise_labels == 1, pairwise_labels, torch.tensor(0, device=device))
    torch.cuda.empty_cache()

    return span_embeddings, first, second, pairwise_labels





if __name__ == '__main__':
    config = pyhocon.ConfigFactory.parse_file(args.config_model_file)
    fix_seed(config)

    if not os.path.exists(config['save_path']):
        os.makedirs(config['save_path'])

    logger = create_logger(config, create_file=True)
    logger.info(pyhocon.HOCONConverter.convert(config, "hocon"))


    if torch.cuda.is_available():
        device = 'cuda:{}'.format(config['gpu_num'])
        torch.cuda.set_device(config['gpu_num'])
    else:
        device = 'cpu'


    # read and tokenize data
    roberta_tokenizer = RobertaTokenizer.from_pretrained(config['roberta_model'])
    sp_raw_text, sp_event_mentions, sp_entity_mentions, sp_tokens = [], [], [], []
    for sp in ['train', 'dev']:
        logger.info('Processing {} set'.format(sp))
        with open(os.path.join(args.data_folder, sp + '_entities.json'), 'r') as f:
            sp_entity_mentions.append(json.load(f))

        with open(os.path.join(args.data_folder, sp + '_events.json'), 'r') as f:
            sp_event_mentions.append(json.load(f))

        with open(os.path.join(args.data_folder, sp + '.json'), 'r') as f:
            ecb_texts = json.load(f)

        if sp == 'train':
            ecb_texts_by_topic = separate_docs_into_topics(ecb_texts, subtopic=True)
        else:
            ecb_texts_by_topic = separate_docs_into_topics(ecb_texts, subtopic=True)
        sp_raw_text.append(ecb_texts_by_topic)
        tokens = tokenize_set(ecb_texts_by_topic, roberta_tokenizer)
        sp_tokens.append(tokens)


    # labels
    logger.info('Get labels')
    event_labels = get_dict_labels(sp_event_mentions)
    entity_labels = get_dict_labels(sp_entity_mentions)


    # Model initiation
    logger.info('Init models')
    bert_model = RobertaModel.from_pretrained(config['roberta_model']).to(device)
    bert_model_hidden_size = bert_model.config.hidden_size

    span_repr = SpanEmbedder(config, bert_model_hidden_size, device).to(device)
    span_scorer = SpanScorer(config, bert_model_hidden_size).to(device)

    if config['training_method'] in ('pipeline', 'fine_tune'):
        span_repr.load_state_dict(torch.load(config['span_repr_path'], map_location=device))
        span_scorer.load_state_dict(torch.load(config['span_scorer_path'], map_location=device))

    span_repr.eval()
    span_scorer.eval()
    pairwise_model = SimplePairWiseClassifier(config, bert_model_hidden_size).to(device)

    models = [pairwise_model]
    if config['training_method'] in ('fine_tune', 'e2e'):
        models.append(span_repr)
        models.append(span_scorer)
    optimizer = get_optimizer(config, models)
    criterion = get_loss_function(config)


    logger.info('Number of parameters of mention extractor: {}'.format(
        count_parameters(span_repr) + count_parameters(span_scorer)))
    logger.info('Number of parameters of the pairwise classifier: {}'.format(
        count_parameters(pairwise_model)))


    training_set = sp_tokens[0]
    all_topics, topic_list_of_docs, topic_origin_tokens, topic_bert_tokens, topic_start_end_bert = training_set
    logger.info('Number of topics: {}'.format(len(all_topics)))
    max_dev = (0, None)

    for epoch in range(config['epochs']):
        logger.info('Epoch: {}'.format(epoch))
        pairwise_model.train()
        if config['training_method'] in ('fine_tune', 'e2e'):
            span_repr.train()
            span_scorer.train()


        accumulate_loss = 0
        list_of_topics = shuffle(list(range(len(all_topics))))
        for t in tqdm(list_of_topics):
            topic = all_topics[t]
            span_embeddings, first, second, pairwise_labels = \
                get_all_pairs_from_topic(config, bert_model, span_repr, span_scorer,
                                         topic_list_of_docs[t], topic_origin_tokens[t], topic_bert_tokens[t],
                                         topic_start_end_bert[t], event_labels, entity_labels)
            loss = train_pairwise_classifier(config, pairwise_model, span_repr, span_scorer, span_embeddings, first,
                                                   second, pairwise_labels, config['batch_size'], criterion, optimizer)
            torch.cuda.empty_cache()
            accumulate_loss += loss
            # logger.info('Number of positive pairs: {}/{}'.format(len(pairwise_labels.nonzero()), len(pairwise_labels)))
        logger.info('Accumulate loss: {}'.format(accumulate_loss))



        logger.info('Evaluate on the dev set')
        dev_all_topics, dev_topic_list_of_docs, dev_topic_origin_tokens, dev_topic_bert_tokens, dev_topic_start_end_bert = sp_tokens[1]
        all_scores, all_labels = [], []
        span_repr.eval()
        span_scorer.eval()
        pairwise_model.eval()

        for t, topic in enumerate(tqdm(dev_all_topics)):
            span_embeddings, first, second, pairwise_labels = \
                get_all_pairs_from_topic(config, bert_model, span_repr, span_scorer,
                                         dev_topic_list_of_docs[t], dev_topic_origin_tokens[t], dev_topic_bert_tokens[t],
                                         dev_topic_start_end_bert[t], event_labels, entity_labels)

            start_end_embeddings, continuous_embeddings, width = span_embeddings
            width = width.to(device)
            with torch.no_grad():
                g1 = span_repr(start_end_embeddings[first],
                                          [continuous_embeddings[i] for i in first],
                                          width[first])
                g2 = span_repr(start_end_embeddings[second],
                                          [continuous_embeddings[i] for i in second],
                                          width[second])

                scores = pairwise_model(g1, g2)
                if config['training_method'] in ('fine_tune', 'e2e'):
                    g1_score = span_scorer(g1)
                    g2_score = span_scorer(g2)
                    scores += g1_score + g2_score

            all_scores.extend(scores.squeeze(1))
            all_labels.extend(pairwise_labels.to(torch.int))

        all_labels = torch.stack(all_labels)
        all_scores = torch.stack(all_scores)


        strict_preds = (all_scores > 0).to(torch.int)
        eval = Evaluation(strict_preds, all_labels.to(device))
        logger.info('Number of predictions: {}/{}'.format(strict_preds.sum(), len(strict_preds)))
        logger.info('Number of positive pairs: {}/{}'.format(len((all_labels == 1).nonzero()),
                                                             len(all_labels)))
        logger.info('Strict - Recall: {}, Precision: {}, F1: {}'.format(eval.get_recall(),
                                                                        eval.get_precision(), eval.get_f1()))

        if eval.get_f1() > max_dev[0]:
            max_dev = (eval.get_f1(), epoch)

        torch.save(span_repr.state_dict(), os.path.join(config['model_path'], 'span_repr_{}'.format(epoch)))
        torch.save(span_scorer.state_dict(), os.path.join(config['model_path'], 'span_scorer_{}'.format(epoch)))
        torch.save(pairwise_model.state_dict(), os.path.join(config['model_path'], 'pairwise_scorer_{}'.format(epoch)))


    logger.info('Best Performance: {}'.format(max_dev))

    user = 'gpus.experiment@gmail.com'
    pwd = 'Gpusexperiments'
    recipient = 'arie.cattan@gmail.com'
    subject = 'GPUs experiments are done'
    message = 'Best Dev F1: {}'.format(max_dev) + '\n\n' + str(pyhocon.HOCONConverter.convert(config, "hocon"))

    send_email(user, pwd, recipient, subject, message)