import argparse
import json
import pyhocon
from sklearn.utils import shuffle
from models import SpanEmbedder, SpanScorer
from transformers import RobertaTokenizer, RobertaModel
from evaluator import Evaluation
from tqdm import tqdm

from model_utils import *
from utils import *



parser = argparse.ArgumentParser()
parser.add_argument('--data_folder', type=str, default='data/ecb/mentions')
parser.add_argument('--config_model_file', type=str, default='configs/config_span_scorer.json')
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







def get_span_data_from_topic(config, bert_model, list_of_docs, docs_original_tokens, bert_tokens,
                                  docs_bert_start_end, event_labels, entity_labels):
    docs_embeddings, docs_length = pad_and_read_bert(bert_tokens, bert_model)
    span_meta_data, span_embeddings, num_of_tokens = get_all_candidate_from_topic(
        config, list_of_docs, docs_original_tokens, docs_bert_start_end,
        docs_embeddings, docs_length)
    doc_id, sentence_id, start, end = span_meta_data
    event_labels, entity_labels = get_candidate_labels(doc_id, start, end, event_labels,
                                                       entity_labels)

    labels = event_labels if config['is_event'] else entity_labels
    mention_labels = torch.zeros(labels.shape, device=device)
    mention_labels[labels.nonzero().squeeze(1)] = 1

    return span_meta_data, span_embeddings, mention_labels, num_of_tokens



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
    roberta_tokenizer = RobertaTokenizer.from_pretrained(config['roberta_model'], add_special_tokens=True)
    sp_raw_text, sp_event_mentions, sp_entity_mentions, sp_tokens = [], [], [], []
    for sp in ['train', 'dev']:
        logger.info('Processing {} set'.format(sp))
        with open(os.path.join(args.data_folder, sp + '_entities.json'), 'r') as f:
            sp_entity_mentions.append(json.load(f))

        with open(os.path.join(args.data_folder, sp + '_events.json'), 'r') as f:
            sp_event_mentions.append(json.load(f))

        with open(os.path.join(args.data_folder, sp + '.json'), 'r') as f:
            ecb_texts = json.load(f)
        ecb_texts_by_topic = separate_docs_into_topics(ecb_texts, subtopic=False)
        sp_raw_text.append(ecb_texts_by_topic)
        tokens = tokenize_set(ecb_texts_by_topic, roberta_tokenizer)
        sp_tokens.append(tokens)


    # labels
    logger.info('Get labels')
    event_labels = get_dict_labels(sp_event_mentions)
    entity_labels = get_dict_labels(sp_entity_mentions)


    # Mention extractor configuration
    logger.info('Init models')
    bert_model = RobertaModel.from_pretrained(config['roberta_model']).to(device)
    bert_model_hidden_size = bert_model.config.hidden_size
    span_repr = SpanEmbedder(config, bert_model_hidden_size, device).to(device)
    span_scorer = SpanScorer(config, bert_model_hidden_size).to(device)
    optimizer = get_optimizer(config, [span_scorer, span_repr])
    criterion = get_loss_function(config)


    logger.info('Number of parameters of mention extractor: {}'.format(
        count_parameters(span_repr) + count_parameters(span_scorer)))

    if config['is_event']:
        eval_range = [0.2, 0.25, 0.3]
        span_repr_path = os.path.join(config['model_path'], 'event_span_repr_{}'.format(config['exp_num']))
        span_scorer_path = os.path.join(config['model_path'], 'event_span_scorer_{}'.format(config['exp_num']))
        logger.info('Train event mention detection')
    else:
        eval_range = [0.2, 0.25, 0.3, 0.4]
        span_repr_path = os.path.join(config['model_path'], 'entity_span_repr_{}'.format(config['exp_num']))
        span_scorer_path = os.path.join(config['model_path'], 'entity_span_scorer_{}'.format(config['exp_num']))
        logger.info('Train entity mention detection')


    training_set = sp_tokens[0]
    all_topics, topic_list_of_docs, topic_origin_tokens, topic_bert_tokens, topic_start_end_bert = training_set
    logger.info('Number of topics: {}'.format(len(all_topics)))
    max_dev = (0, None)

    for epoch in range(config['epochs']):
        logger.info('Epoch: {}'.format(epoch))
        list_of_topics = shuffle(list(range(len(all_topics))))
        total_loss = 0
        span_repr.train()
        span_scorer.train()

        for t in tqdm(list_of_topics):
            topic = all_topics[t]
            # logger.info('Training on topic {}'.format(topic))
            span_meta_data, span_embeddings, mention_labels, num_of_tokens = get_span_data_from_topic(config, bert_model,
                                                                                       topic_list_of_docs[t],
                                                                                       topic_origin_tokens[t],
                                                                                       topic_bert_tokens[t],
                                                                                       topic_start_end_bert[t],
                                                                                       event_labels,
                                                                                       entity_labels)

            topic_start_end_embeddings, topic_continuous_embeddings, topic_width = span_embeddings
            epoch_loss = train_topic_mention_extractor(span_repr, span_scorer, topic_start_end_embeddings,
                                          topic_continuous_embeddings, topic_width.to(device),
                                          mention_labels, config['batch_size'], criterion, optimizer)
            total_loss += epoch_loss
            torch.cuda.empty_cache()


        logger.info('Evaluate on the dev set')
        dev_all_topics, dev_topic_list_of_docs, dev_topic_origin_tokens, \
        dev_topic_bert_tokens, dev_topic_start_end_bert = sp_tokens[1]
        span_repr.eval()
        span_scorer.eval()

        all_scores, all_labels = [], []
        list_of_dev_topics = list(range(len(dev_all_topics)))
        dev_num_of_tokens = 0
        for t in tqdm(list_of_dev_topics):
            span_meta_data, span_embeddings, mention_labels, num_of_tokens = get_span_data_from_topic(config, bert_model,
                                                                                       dev_topic_list_of_docs[t],
                                                                                       dev_topic_origin_tokens[t],
                                                                                       dev_topic_bert_tokens[t],
                                                                                       dev_topic_start_end_bert[t],
                                                                                       event_labels,
                                                                                       entity_labels)

            all_labels.extend(mention_labels)
            dev_num_of_tokens += num_of_tokens
            topic_start_end_embeddings, topic_continuous_embeddings, topic_width = span_embeddings
            with torch.no_grad():
                span_emb = span_repr(topic_start_end_embeddings, topic_continuous_embeddings, topic_width.to(device))
                span_score = span_scorer(span_emb)
            all_scores.extend(span_score.squeeze(1))


        all_scores = torch.stack(all_scores)
        all_labels = torch.stack(all_labels)


        strict_preds = (all_scores > 0).to(torch.int)
        eval = Evaluation(strict_preds, all_labels)
        logger.info(
            'Recall: {}, Precision: {}, F1: {}'.format(eval.get_recall(),
                                                       eval.get_precision(), eval.get_f1()))

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