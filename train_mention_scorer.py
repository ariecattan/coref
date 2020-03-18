import argparse
import json
import pyhocon
from sklearn.utils import shuffle
from models import MentionExtractor, SimplePairWiseClassifier
from transformers import RobertaTokenizer, RobertaModel
import torch.nn as nn
import torch.optim as optim
from evaluator import Evaluation
from utils import *



parser = argparse.ArgumentParser()
parser.add_argument('--data_folder', type=str, default='data/ecb/mentions')
parser.add_argument('--config_model_file', type=str, default='config_mention_extractor.json')
parser.add_argument('--entity_mention_extractor_path', type=str, default='models/with_att/entity_mention_extractor_5')
parser.add_argument('--event_mention_extractor_path', type=str, default='models/with_att/event_mention_extractor_5')
args = parser.parse_args()


is_training = True



def train_topic_mention_extractor(model, start_end, continuous_embeddings,
                                width, labels, batch_size, criterion, optimizer):
    model.train()
    for i in range(0, len(width), batch_size):
        batch_start_end = start_end[i:i+batch_size]
        batch_width = width[i:i+batch_size]
        batch_continuous_embeddings = continuous_embeddings[i:i+batch_size]

        batch_labels = labels[i:i+batch_size]
        optimizer.zero_grad()
        _, preds = model(batch_start_end, batch_continuous_embeddings, batch_width)
        loss = criterion(preds.squeeze(1), batch_labels)
        loss.backward()
        optimizer.step()





def prepare_data(sp, bert_model, config, device):
    all_topics, topic_list_of_docs, topic_origin_tokens, topic_bert_tokens, topic_start_end_bert = sp

    start_end_embeddings = []
    continuous_embeddings = []
    width = []
    num_tokens = 0
    meta_data_0, meta_data_1, meta_data_2, meta_data_3 = [], [], [], []

    for t, topic in enumerate(all_topics):
        list_of_docs = topic_list_of_docs[t]
        docs_original_tokens = topic_origin_tokens[t]
        bert_tokens = topic_bert_tokens[t]
        docs_bert_start_end = topic_start_end_bert[t]

        docs_embeddings, docs_length = pad_and_read_bert(bert_tokens, bert_model, device)
        span_meta_data, span_embeddings, num_of_tokens = get_all_candidate_from_topic(
            config, device, list_of_docs, docs_original_tokens, docs_bert_start_end,
            docs_embeddings, docs_length, is_training=is_training)

        meta_data_0.append(span_meta_data[0])
        meta_data_1.append(span_meta_data[1])
        meta_data_2.append(span_meta_data[2])
        meta_data_3.append(span_meta_data[3])

        num_tokens += num_of_tokens
        topic_start_end_embeddings, topic_continuous_embeddings, topic_width = span_embeddings

        start_end_embeddings.append(topic_start_end_embeddings)
        continuous_embeddings.append(topic_continuous_embeddings)
        width.append(topic_width)


    return [meta_data_0, meta_data_1, meta_data_2, meta_data_3], \
           [start_end_embeddings, continuous_embeddings, width],\
           num_tokens





def get_span_label(dev_candidate_meta_data, device, event_labels, entity_labels):
    dev_doc_id, dev_sentence_id, dev_start, dev_end = dev_candidate_meta_data
    dev_event_labels, dev_entity_labels = [], []
    for i in range(len(dev_doc_id)):
        dev_topic_event_labels, dev_topic_entity_labels = get_candidate_labels(device, dev_doc_id[i], dev_start[i],
                                                                               dev_end[i], event_labels, entity_labels)
        dev_event_labels.extend(dev_topic_event_labels)
        dev_entity_labels.extend(dev_topic_entity_labels)
    dev_entity_labels = torch.stack(dev_entity_labels).to(device)
    dev_event_labels = torch.stack(dev_event_labels).to(device)

    dev_entity_mention_labels = torch.zeros(len(dev_entity_labels), device=device)
    dev_entity_mention_labels[dev_entity_labels.nonzero().squeeze(1)] = 1
    dev_event_mention_labels = torch.zeros(len(dev_event_labels), device=device)
    dev_event_mention_labels[dev_event_labels.nonzero().squeeze(1)] = 1

    return dev_event_mention_labels, dev_entity_mention_labels




if __name__ == '__main__':
    config = pyhocon.ConfigFactory.parse_file(args.config_model_file)
    fix_seed(config)
    use_gold_mentions = config['use_gold_mentions']

    if not os.path.exists(config['save_path']):
        os.makedirs(config['save_path'])

    logger = create_logger(config, create_file=True)
    logger.info(pyhocon.HOCONConverter.convert(config, "hocon"))


    if torch.cuda.is_available():
        device = 'cuda:{}'.format(config['gpu_num'])
        torch.cuda.set_device(config['gpu_num'])
    else:
        device = 'cpu'

    dev_device = 'cuda:{}'.format(config['dev_gpu_num']) if torch.cuda.is_available() else 'cpu'


    # read and tokenize data
    roberta_tokenizer = RobertaTokenizer.from_pretrained(config['roberta_model'], add_special_tokens=True)
    sp_raw_text, sp_event_mentions, sp_entity_mentions, sp_tokens = [], [], [], []
    for sp in ['train', 'dev', 'test']:
        logger.info('Processing {} set'.format(sp))
        with open(os.path.join(args.data_folder, sp + '_entities.json'), 'r') as f:
            sp_entity_mentions.append(json.load(f))

        with open(os.path.join(args.data_folder, sp + '_events.json'), 'r') as f:
            sp_event_mentions.append(json.load(f))

        with open(os.path.join(args.data_folder, sp + '.json'), 'r') as f:
            ecb_texts = json.load(f)
        ecb_texts_by_topic = separate_docs_into_topics(ecb_texts)
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
    bert_model_hidden_size = 768 if 'base' in config['roberta_model'] else 1024
    event_mention_extractor = MentionExtractor(config, bert_model_hidden_size, device).to(device)
    event_optimizer = optim.Adam(event_mention_extractor.parameters(), lr=config['learning_rate'],
                                 weight_decay=config['weight_decay'])

    entity_mention_extractor = MentionExtractor(config, bert_model_hidden_size, device).to(device)
    entity_optimizer = optim.Adam(entity_mention_extractor.parameters(), lr=config['learning_rate'],
                                  weight_decay=config['weight_decay'])

    criterion = nn.BCEWithLogitsLoss()

    logger.info('Number of parameters of mention extractor: {}'.format(count_parameters(event_mention_extractor)))


    # Prepare dev data and labels
    logger.info('Prepare dev data')
    dev_candidate_data = prepare_data(sp_tokens[1], bert_model, config, dev_device)
    dev_event_mention_labels, dev_entity_mention_labels = get_span_label(dev_candidate_data[0], dev_device, event_labels, entity_labels)
    dev_num_tokens = dev_candidate_data[-1]


    if config['is_event']:
        mention_extractor = event_mention_extractor
        optimizer = event_optimizer
        eval_range = [0.2, 0.25, 0.3]
        dev_labels = dev_event_mention_labels
        mention_extractor_path = args.event_mention_extractor_path
        logger.info('Train event mention detection')
    else:
        mention_extractor = entity_mention_extractor
        optimizer = entity_optimizer
        eval_range = [0.2, 0.25, 0.3, 0.4]
        dev_labels = dev_entity_mention_labels
        mention_extractor_path = args.entity_mention_extractor_path
        logger.info('Train entity mention detection')



    training_set = sp_tokens[0]
    all_topics, topic_list_of_docs, topic_origin_tokens, topic_bert_tokens, topic_start_end_bert = training_set
    logger.info('Number of topics: {}'.format(len(all_topics)))
    max_dev = (0, None)

    event_mention_extractor.eval()

    for epoch in range(config['epochs']):
        logger.info('Epoch: {}'.format(epoch))
        accumulate_loss = 0
        positive_pairs = 0
        list_of_topics = shuffle(list(range(len(all_topics))))
        for t in list_of_topics:
            topic = all_topics[t]
            logger.info('Training on topic {}'.format(topic))
            list_of_docs = topic_list_of_docs[t]
            docs_original_tokens = topic_origin_tokens[t]
            bert_tokens = topic_bert_tokens[t]
            docs_bert_start_end = topic_start_end_bert[t]

            docs_embeddings, docs_length = pad_and_read_bert(bert_tokens, bert_model, device)
            span_meta_data, span_embeddings, num_of_tokens = get_all_candidate_from_topic(
                config, device, list_of_docs, docs_original_tokens, docs_bert_start_end, docs_embeddings, docs_length,
                is_training)
            topic_start_end_embeddings, topic_continuous_embeddings, topic_width = span_embeddings
            torch.cuda.empty_cache()

            doc_id, sentence_id, start, end = span_meta_data
            train_event_labels, train_entity_labels = get_candidate_labels(device, doc_id, start, end, event_labels,
                                                                           entity_labels)

            train_labels = train_event_labels if config['is_event'] else train_entity_labels
            mention_labels = torch.zeros(train_labels.shape, device=device)
            mention_labels[train_labels.nonzero().squeeze(1)] = 1

            train_topic_mention_extractor(mention_extractor, topic_start_end_embeddings,
                                          topic_continuous_embeddings,
                                          topic_width, mention_labels, config['batch_size'], criterion, optimizer)
            torch.cuda.empty_cache()


        logger.info('Evaluate on the dev set')
        all_scores = evaluate_model(dev_candidate_data[1], mention_extractor)
        strict_preds = (all_scores > 0).to(torch.int)
        eval = Evaluation(strict_preds, dev_labels)
        logger.info(
            'Recall: {}, Precision: {}, F1: {}'.format(eval.get_recall(), eval.get_precision(), eval.get_f1()))

        for k in eval_range:
            s, i = torch.topk(all_scores, int(k * dev_num_tokens), sorted=False)
            rank_preds = torch.zeros(len(all_scores), device=dev_device)
            rank_preds[i] = 1
            eval = Evaluation(rank_preds, dev_labels)
            recall = eval.get_recall()
            if recall > max_dev[0]:
                max_dev = (recall, epoch)
                # torczh.save(mention_extractor.state_dict(), mention_extractor_path)

            logger.info(
                'K = {}, Recall: {}, Precision: {}, F1: {}'.format(k, eval.get_recall(), eval.get_precision(),
                                                                   eval.get_f1()))



    logger.info('Best Performance: {}'.format(max_dev))