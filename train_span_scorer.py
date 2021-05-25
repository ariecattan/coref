import argparse
import pyhocon
from sklearn.utils import shuffle
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

from evaluator import Evaluation
from models import SpanEmbedder, SpanScorer
from model_utils import *
from utils import *


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/config_span_scorer.json')
args = parser.parse_args()




def train_topic_mention_extractor(span_repr, span_scorer, start_end, continuous_embeddings,
                                width, labels, batch_size, criterion, optimizer):
    accumulate_loss = 0
    idx = list(range(len(width)))
    for i in range(0, len(width), batch_size):
        indices = idx[i:i+batch_size]
        batch_start_end = start_end[indices]
        batch_width = width[indices]
        batch_continuous_embeddings = [continuous_embeddings[k] for k in indices]
        batch_labels = labels[i:i + batch_size]
        optimizer.zero_grad()
        span = span_repr(batch_start_end, batch_continuous_embeddings, batch_width)
        scores = span_scorer(span)
        loss = criterion(scores.squeeze(1), batch_labels)
        loss.backward()
        accumulate_loss += loss.item()
        optimizer.step()
    return accumulate_loss







def get_span_data_from_topic(config, bert_model, data, topic_num):
    docs_embeddings, docs_length = pad_and_read_bert(data.topics_bert_tokens[topic_num], bert_model)
    span_meta_data, span_embeddings, num_of_tokens = get_all_candidate_from_topic(
        config, data, topic_num, docs_embeddings, docs_length)
    doc_id, sentence_id, start, end = span_meta_data
    labels = data.get_candidate_labels(doc_id, start, end)
    mention_labels = torch.zeros(labels.shape, device=device)
    mention_labels[labels.nonzero().squeeze(1)] = 1

    return span_meta_data, span_embeddings, mention_labels, num_of_tokens





if __name__ == '__main__':
    config = pyhocon.ConfigFactory.parse_file(args.config)
    fix_seed(config)

    logger = create_logger(config, create_file=True)
    logger.info(pyhocon.HOCONConverter.convert(config, "hocon"))
    create_folder(config['model_path'])

    if torch.cuda.is_available():
        device = 'cuda:{}'.format(config['gpu_num'])
        torch.cuda.set_device(config['gpu_num'])
    else:
        device = 'cpu'


    # read and tokenize data
    bert_tokenizer = AutoTokenizer.from_pretrained(config['bert_model'], add_special_tokens=True)
    training_set = create_corpus(config, bert_tokenizer, 'train')
    dev_set = create_corpus(config, bert_tokenizer, 'dev')



    # Mention extractor configuration
    logger.info('Init models')
    bert_model = AutoModel.from_pretrained(config['bert_model']).to(device)
    config['bert_hidden_size'] = bert_model.config.hidden_size
    span_repr = SpanEmbedder(config, device).to(device)
    span_scorer = SpanScorer(config).to(device)
    optimizer = get_optimizer(config, [span_scorer, span_repr])
    criterion = get_loss_function(config)


    logger.info('Number of parameters of mention extractor: {}'.format(
        count_parameters(span_repr) + count_parameters(span_scorer)))

    span_repr_path = os.path.join(config['model_path'],
                                  '{}_span_repr_{}'.format(config['mention_type'], config['exp_num']))
    span_scorer_path = os.path.join(config['model_path'],
                                    '{}_span_scorer_{}'.format(config['mention_type'], config['exp_num']))


    logger.info('Number of topics: {}'.format(len(training_set.topic_list)))
    max_dev = (0, None)

    for epoch in range(config['epochs']):
        logger.info('Epoch: {}'.format(epoch))

        span_repr.train()
        span_scorer.train()

        list_of_topics = shuffle(list(range(len(training_set.topic_list))))
        accumulate_loss = 0

        for topic_num in tqdm(list_of_topics):
            topic = training_set.topic_list[topic_num]
            span_meta_data, span_embeddings, mention_labels, num_of_tokens = \
                get_span_data_from_topic(config, bert_model, training_set, topic_num)

            topic_start_end_embeddings, topic_continuous_embeddings, topic_width = span_embeddings
            epoch_loss = train_topic_mention_extractor(span_repr, span_scorer, topic_start_end_embeddings,
                                          topic_continuous_embeddings, topic_width.to(device),
                                          mention_labels, config['batch_size'], criterion, optimizer)
            accumulate_loss += epoch_loss
            torch.cuda.empty_cache()

        logger.info('Accumulate loss: {}'.format(accumulate_loss))


        logger.info('Evaluate on the dev set')

        span_repr.eval()
        span_scorer.eval()

        all_scores, all_labels = [], []
        dev_num_of_tokens = 0
        for topic_num, topic in enumerate(tqdm(dev_set.topic_list)):
            span_meta_data, span_embeddings, mention_labels, num_of_tokens = \
                get_span_data_from_topic(config, bert_model, dev_set, topic_num)

            all_labels.extend(mention_labels)
            dev_num_of_tokens += num_of_tokens
            topic_start_end_embeddings, topic_continuous_embeddings, topic_width = span_embeddings
            with torch.no_grad():
                span_emb = span_repr(topic_start_end_embeddings, topic_continuous_embeddings,
                                     topic_width.to(device))
                span_score = span_scorer(span_emb)
            all_scores.extend(span_score.squeeze(1))


        all_scores = torch.stack(all_scores)
        all_labels = torch.stack(all_labels)


        strict_preds = (all_scores > 0).to(torch.int)
        eval = Evaluation(strict_preds, all_labels)
        logger.info(
            'Recall: {}, Precision: {}, F1: {}'.format(eval.get_recall(),
                                                       eval.get_precision(), eval.get_f1()))

        if config.exact:
            if eval.get_f1() > max_dev[0]:
                max_dev = (eval.get_f1(), epoch)
                torch.save(span_repr.state_dict(), span_repr_path)
                torch.save(span_scorer.state_dict(), span_scorer_path)

        else:
            eval_range = [0.2, 0.25, 0.3] if config['mention_type'] == 'events' else [0.2, 0.25, 0.3, 0.4, 0.45]
            for k in eval_range:
                s, i = torch.topk(all_scores, int(k * dev_num_of_tokens), sorted=False)
                rank_preds = torch.zeros(len(all_scores), device=device)
                rank_preds[i] = 1
                eval = Evaluation(rank_preds, all_labels)
                recall = eval.get_recall()
                if recall > max_dev[0]:
                    max_dev = (recall, epoch)
                    torch.save(span_repr.state_dict(), span_repr_path)
                    torch.save(span_scorer.state_dict(), span_scorer_path)
                logger.info(
                    'K = {}, Recall: {}, Precision: {}, F1: {}'.format(k, eval.get_recall(), eval.get_precision(),
                                                                       eval.get_f1()))



    logger.info('Best Performance: {}'.format(max_dev))