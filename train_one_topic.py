import argparse
import json
import pyhocon
from sklearn.utils import shuffle
from mention_extractor import MentionExtractor, SimplePairWiseClassifier
from transformers import RobertaTokenizer, RobertaModel
import torch.nn as nn
import torch.optim as optim
from itertools import compress, combinations
from evaluator import Evaluation
from utils import *
from copy import deepcopy
from sklearn.cluster import *


parser = argparse.ArgumentParser()
parser.add_argument('--data_folder', type=str, default='data/ecb/mentions')
parser.add_argument('--config_model_file', type=str, default='config_mention_extractor.json')
parser.add_argument('--train_mention_extractor', type=bool, default=False)
parser.add_argument('--entity_mention_extractor_path', type=str, default='models/without_att/entity_mention_extractor_5')
parser.add_argument('--event_mention_extractor_path', type=str, default='models/without_att/event_mention_extractor_5')
parser.add_argument('--pairwise_path', type=str, default='models/without_att/pairwise_model_gold_mentions')
args = parser.parse_args()


is_training = True


def get_candidate_mention_label(candidate_mentions, doc_gold_mentions):
    gold_mention_spans = {(min(m['tokens_ids']), max(m['tokens_ids'])): m['cluster_id']
                          for m in doc_gold_mentions}
    labels = []
    candidate_starts, candidate_ends = candidate_mentions
    for i, (s, e) in enumerate(zip(candidate_starts.tolist(), candidate_ends.tolist())):
        cluster_id = gold_mention_spans.get((s, e), 0)
        labels.append(cluster_id)

    return torch.tensor(labels)


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


def predict_mention_extractor(model, start_end, continuous_embeddings, width):
    model.eval()
    with torch.no_grad():
        _, scores = model(start_end, continuous_embeddings, width)
        scores = scores.squeeze(1)
    return scores


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



def evaluate_model(dev_span_embeddings, model):
    all_scores = []
    start_end_embeddings, continuous_embeddings, width = dev_span_embeddings
    for i in range(len(start_end_embeddings)):
        scores = predict_mention_extractor(model, start_end_embeddings[i], continuous_embeddings[i], width[i])
        all_scores.extend(scores)

    return torch.stack(all_scores)


def batch_train_pairwise_classifier(model, first, second, labels, batch_size, criterion, optimizer):
    model.train()
    accumulate_loss = 0
    for i in range(0, len(first), batch_size):
        batch_first = first[i:i+batch_size]
        batch_second = second[i:i+batch_size]
        batch_labels = labels[i:i+batch_size].to(torch.float)
        optimizer.zero_grad()
        scores = model(batch_first, batch_second)
        loss = criterion(scores.squeeze(1), batch_labels)
        accumulate_loss += loss.item()
        loss.backward(retain_graph=True)
        optimizer.step()

    return accumulate_loss




def train_pairwise_classifier(model, first, second, labels, batch_size, criterion, optimizer):
    model.train()
    optimizer.zero_grad()
    scores = model(first, second)
    loss = criterion(scores.squeeze(1), labels.to(torch.float))
    loss.backward()
    optimizer.step()
    torch.cuda.empty_cache()

    return loss.item()



if __name__ == '__main__':
    config = pyhocon.ConfigFactory.parse_file(args.config_model_file)
    fix_seed(config)

    use_gold_mentions = config['use_gold_mentions']
    e2e = config['e2e']

    if not os.path.exists(config['save_path']):
        os.makedirs(config['save_path'])

    logger = create_logger(config, create_file=True)
    logger.info(pyhocon.HOCONConverter.convert(config, "hocon"))

    # read and tokenize data

    if torch.cuda.is_available():
        device = 'cuda:{}'.format(config['gpu_num'])
        torch.cuda.set_device(config['gpu_num'])
    else:
        device = 'cpu'

    dev_device = 'cuda:{}'.format(config['dev_gpu_num']) if torch.cuda.is_available() else 'cpu'

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
    dev_bert_model = deepcopy(bert_model).to(dev_device)
    bert_model_hidden_size = 768 if 'base' in config['roberta_model'] else 1024
    event_mention_extractor = MentionExtractor(config, bert_model_hidden_size, device).to(device)
    event_mention_extractor_clone = MentionExtractor(config, bert_model_hidden_size,
                                               dev_device).to(dev_device)
    event_optimizer = optim.Adam(event_mention_extractor.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])


    entity_mention_extractor = MentionExtractor(config, bert_model_hidden_size, device).to(device)
    entity_mention_extractor_clone = MentionExtractor(config, bert_model_hidden_size,
                                                     dev_device).to(dev_device)
    entity_optimizer = optim.Adam(entity_mention_extractor.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])


    pairwise_classifier = SimplePairWiseClassifier(config, bert_model_hidden_size).to(device)
    pairwise_classifier_clone = SimplePairWiseClassifier(config, bert_model_hidden_size).to(dev_device)
    pairwise_optimizer = optim.Adam(pairwise_classifier.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

    criterion = nn.BCEWithLogitsLoss()

    logger.info('Number of parameters of mention extractor: {}'.format(count_parameters(event_mention_extractor)))
    logger.info('Number of parameters of pairwise scorer: {}'.format(count_parameters(pairwise_classifier)))



    if not args.train_mention_extractor and not e2e:
        event_mention_extractor.load_state_dict(torch.load(args.event_mention_extractor_path))
        entity_mention_extractor.load_state_dict(torch.load(args.entity_mention_extractor_path))




    # Prepare dev data and labels
    logger.info('Prepare dev data')
    dev_candidate_data = prepare_data(sp_tokens[1], dev_bert_model, config, dev_device)
    dev_doc_id, dev_sentence_id, dev_start, dev_end = dev_candidate_data[0]
    dev_event_labels, dev_entity_labels = [], []
    for i in range(len(dev_doc_id)):
        dev_topic_event_labels, dev_topic_entity_labels = get_candidate_labels(dev_device, dev_doc_id[i], dev_start[i], dev_end[i], event_labels, entity_labels)
        dev_event_labels.extend(dev_topic_event_labels)
        dev_entity_labels.extend(dev_topic_entity_labels)
    dev_entity_labels = torch.stack(dev_entity_labels).to(dev_device)
    dev_event_labels = torch.stack(dev_event_labels).to(dev_device)
    dev_num_tokens = dev_candidate_data[-1]

    dev_entity_mention_labels = torch.zeros(len(dev_entity_labels), device=dev_device)
    dev_entity_mention_labels[dev_entity_labels.nonzero().squeeze(1)] = 1
    dev_event_mention_labels = torch.zeros(len(dev_event_labels), device=dev_device)
    dev_event_mention_labels[dev_event_labels.nonzero().squeeze(1)] = 1


    # For mention extractor
    if config['is_event']:
        mention_extractor = event_mention_extractor
        mention_extractor_clone = event_mention_extractor_clone
        optimizer = event_optimizer
        eval_range = [0.2, 0.25, 0.3]
        dev_labels = dev_event_mention_labels
        mention_extractor_path = args.event_mention_extractor_path
        logger.info('Train event mention detection')
    else:
        mention_extractor = entity_mention_extractor
        mention_extractor_clone = entity_mention_extractor_clone
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
                config, device, list_of_docs, docs_original_tokens, docs_bert_start_end, docs_embeddings, docs_length, is_training)
            topic_start_end_embeddings, topic_continuous_embeddings, topic_width = span_embeddings
            torch.cuda.empty_cache()

            doc_id, sentence_id, start, end = span_meta_data
            train_event_labels, train_entity_labels = get_candidate_labels(device, doc_id, start, end, event_labels,
                                                                           entity_labels)

            if args.train_mention_extractor:
                train_labels = train_event_labels if config['is_event'] else train_entity_labels
                mention_labels = torch.zeros(train_labels.shape, device=device)
                mention_labels[train_labels.nonzero().squeeze(1)] = 1

                train_topic_mention_extractor(mention_extractor, topic_start_end_embeddings, topic_continuous_embeddings,
                                              topic_width, mention_labels, config['batch_size'], criterion, optimizer)
                torch.cuda.empty_cache()




            else:
                with torch.no_grad():
                    event_span_embeddings, event_span_scores = event_mention_extractor(topic_start_end_embeddings,
                                                                                           topic_continuous_embeddings,
                                                                                           topic_width)


                # del docs_embeddings, topic_start_end_embeddings, topic_continuous_embeddings, topic_width

                if use_gold_mentions:
                    event_span_indices = train_event_labels.nonzero().squeeze(1)

                else:
                    event_span_scores, event_span_indices = torch.topk(event_span_scores.squeeze(1), int(0.3 * num_of_tokens), sorted=False)
                    # entity_span_scores, entity_span_indices = torch.topk(entity_span_scores.squeeze(1), int(0.4 * num_of_tokens), sorted=False)
                    event_span_indices, _ = torch.sort(event_span_indices)
                    # entity_span_indices, _ = torch.sort(entity_span_indices)


                event_span_embeddings = event_span_embeddings[event_span_indices]
                train_event_labels = train_event_labels[event_span_indices]
                # entity_span_embeddings = entity_span_embeddings[entity_span_indices]
                # train_entity_labels = train_entity_labels[entity_span_indices]


                torch.cuda.empty_cache()
                first, second = zip(*list(combinations(range(len(event_span_indices)), 2)))
                first = torch.tensor(first)
                second = torch.tensor(second)


                pairwise_labels = (train_event_labels[first] != 0) & (train_event_labels[second] != 0) &\
                                  (train_event_labels[first] == train_event_labels[second])
                logger.info('Number of positive pairs: {}/{}'.format(len(pairwise_labels.nonzero()), len(pairwise_labels)))
                positive_pairs += len(pairwise_labels.nonzero())
                first_embeddings = event_span_embeddings[first]
                second_embeddings = event_span_embeddings[second]
                torch.cuda.empty_cache()



                loss = batch_train_pairwise_classifier(pairwise_classifier, first_embeddings, second_embeddings, pairwise_labels,
                                          config['batch_size'], criterion, pairwise_optimizer)
                torch.cuda.empty_cache()
                accumulate_loss += loss
                # pairwise_scores = torch.mm(event_span_embeddings, event_span_embeddings.T)

                # span_doc, span_sentence, span_origin_start, span_origin_end = span_meta_data
                # event_span_doc = span_doc[event_span_indices]
                # event_span_sentence = span_doc[event_span_indices]

        if not args.train_mention_extractor:
            logger.info('Number of positive pairs: {}'.format(positive_pairs))
            logger.info('Accumulate loss: {}'.format(accumulate_loss))

        logger.info('Evaluate on the dev set')
        if args.train_mention_extractor:
            mention_extractor_clone.load_state_dict(mention_extractor.state_dict())
            all_scores = evaluate_model(dev_candidate_data[1], mention_extractor_clone)
            strict_preds = (all_scores > 0).to(torch.int)

            eval = Evaluation(strict_preds, dev_labels)
            logger.info('Recall: {}, Precision: {}, F1: {}'.format(eval.get_recall(), eval.get_precision(), eval.get_f1()))

            for k in eval_range:
                s, i = torch.topk(all_scores, int(k * dev_num_tokens), sorted=False)
                rank_preds = torch.zeros(len(all_scores), device=dev_device)
                rank_preds[i] = 1
                eval = Evaluation(rank_preds, dev_labels)
                recall = eval.get_recall()
                if recall > max_dev[0]:
                    max_dev = (recall, epoch)
                    torch.save(mention_extractor.state_dict(), mention_extractor_path)

                logger.info('K = {}, Recall: {}, Precision: {}, F1: {}'.format(k, eval.get_recall(), eval.get_precision(),
                                                                         eval.get_f1()))


        else:
            dev_all_topics, dev_topic_list_of_docs, dev_topic_origin_tokens, dev_topic_bert_tokens, dev_topic_start_end_bert = sp_tokens[1]
            all_scores, all_labels = [], []
            event_mention_extractor_clone.load_state_dict(event_mention_extractor.state_dict())
            pairwise_classifier_clone.load_state_dict(pairwise_classifier.state_dict())

            event_mention_extractor_clone.eval()
            pairwise_classifier_clone.eval()

            for t, topic in enumerate(dev_all_topics):
                list_of_docs = dev_topic_list_of_docs[t]
                docs_original_tokens = dev_topic_origin_tokens[t]
                bert_tokens = dev_topic_bert_tokens[t]
                docs_bert_start_end = dev_topic_start_end_bert[t]

                docs_embeddings, docs_length = pad_and_read_bert(bert_tokens, dev_bert_model, dev_device)
                span_meta_data, span_embeddings, num_of_tokens = get_all_candidate_from_topic(
                    config, dev_device, list_of_docs, docs_original_tokens, docs_bert_start_end, docs_embeddings,
                    docs_length, is_training)
                topic_start_end_embeddings, topic_continuous_embeddings, topic_width = span_embeddings
                torch.cuda.empty_cache()

                doc_id, sentence_id, start, end = span_meta_data
                dev_event_labels, dev_entity_labels = get_candidate_labels(dev_device, doc_id, start, end, event_labels,
                                                                               entity_labels)

                with torch.no_grad():
                    event_span_embeddings, event_span_scores = event_mention_extractor_clone(topic_start_end_embeddings,
                                                                                       topic_continuous_embeddings,
                                                                                       topic_width)
                del docs_embeddings, topic_start_end_embeddings, topic_continuous_embeddings, topic_width

                if use_gold_mentions:
                    event_span_indices = dev_event_labels.nonzero().squeeze(1)
                else:
                    event_span_scores, event_span_indices = torch.topk(event_span_scores.squeeze(1),
                                                                   int(0.3 * num_of_tokens), sorted=False)

                event_span_indices, _ = torch.sort(event_span_indices)
                event_span_embeddings = event_span_embeddings[event_span_indices]
                dev_event_labels = dev_event_labels[event_span_indices]
                torch.cuda.empty_cache()

                first, second = zip(*list(combinations(range(len(event_span_indices)), 2)))
                first = torch.tensor(first)
                second = torch.tensor(second)




                pairwise_labels = (dev_event_labels[first] != 0) & (
                            dev_event_labels[first] == dev_event_labels[second])
                first_embeddings = event_span_embeddings[first]
                second_embeddings = event_span_embeddings[second]

                with torch.no_grad():
                    pairwise_scores = pairwise_classifier_clone(first_embeddings, second_embeddings)

                all_scores.extend(pairwise_scores.squeeze(1))
                all_labels.extend(pairwise_labels.to(torch.int))

            all_labels = torch.stack(all_labels)
            all_scores = torch.stack(all_scores)
            strict_preds = (all_scores > 0).to(torch.int)
            eval = Evaluation(strict_preds, all_labels)
            logger.info('Number of positive pairs: {}/{}'.format(len(all_labels.nonzero()), len(all_labels)))
            logger.info('Recall: {}, Precision: {}, F1: {}'.format(eval.get_recall(), eval.get_precision(), eval.get_f1()))

            if eval.get_f1() > max_dev[0]:
                max_dev = (eval.get_f1(), epoch)
                torch.save(pairwise_classifier.state_dict(), args.pairwise_path)

            # s, i = torch.topk(all_scores, int(0.02 * len(all_scores)), sorted=False)
            # rank_preds = torch.zeros(len(all_scores), device=dev_device)
            # rank_preds[i.squeeze(1)] = 1
            # eval = Evaluation(rank_preds, all_labels)
            # logger.info(
            #     'Recall: {}, Precision: {}, F1: {}'.format(eval.get_recall(), eval.get_precision(), eval.get_f1()))
            # clustering = AgglomerativeClustering(n_clusters=None, affinity='precomputed', distance_threshold=0)




    logger.info('Best Performance: {}'.format(max_dev))