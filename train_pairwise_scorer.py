import argparse
import json
import pyhocon
from sklearn.utils import shuffle
from models import SimpleMentionExtractor, MentionExtractor, SimplePairWiseClassifier, GraphPairwiseClassifier
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
parser.add_argument('--config_model_file', type=str, default='configs/config_pairwise.json')
parser.add_argument('--entity_mention_extractor_path', type=str, default='models/without_att/entity_mention_extractor_5')
parser.add_argument('--event_mention_extractor_path', type=str, default='models/without_att/event_mention_extractor_5')
parser.add_argument('--big_mention_extractor_path', type=str, default='models/with_att/event_mention_extractor_5')
parser.add_argument('--pairwise_path', type=str, default='models/with_att/pairwise_model_gold_mentions')
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





def batch_train_pairwise_classifier(model, first, second, labels, batch_size, criterion, optimizer):
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



def create_adj_matrix(span_meta_data, event_span_indices, entity_span_indices, device):
    span_doc, span_sentence, span_start, span_end = span_meta_data
    event_indices = event_span_indices.cpu().numpy()
    entity_indices = entity_span_indices.cpu().numpy()

    event_span_doc, entity_span_doc = span_doc[event_indices], span_doc[entity_indices]
    event_span_sentence, entity_span_sentence = span_sentence[event_indices].numpy(), span_sentence[entity_indices].numpy()
    # event_span_start, entity_span_start = span_start[event_indices], span_start[entity_indices]
    # event_span_end, entity_span_end = span_end[event_indices], span_end[entity_indices]

    event_adj = torch.zeros((len(event_span_indices), len(event_span_indices)))
    entity_adj = torch.zeros((len(event_span_indices), len(entity_span_indices)))


    event_idx = {}
    for i, (doc, sent) in enumerate(zip(event_span_doc, event_span_sentence)):
        if (doc, sent) not in event_idx:
            event_idx[(doc, sent)] = []
        event_idx[(doc, sent)].append(i)

    entity_idx = {}
    for i, (doc, sent) in enumerate(zip(entity_span_doc, entity_span_sentence)):
        if (doc, sent) not in entity_idx:
            entity_idx[(doc, sent)] = []
        entity_idx[(doc, sent)].append(i)


    for i, (doc, sent) in enumerate(zip(event_span_doc, event_span_sentence)):
        events = event_idx[(doc, sent)]
        event_adj[i][events] = 1

        entities = entity_idx[(doc, sent)]
        entity_adj[i][entities] = 1


    return event_adj.to(device), entity_adj.to(device)





if __name__ == '__main__':
    config = pyhocon.ConfigFactory.parse_file(args.config_model_file)
    fix_seed(config)
    use_gold_mentions = config['use_gold_mentions']
    e2e = config['e2e']

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

    if config['use_simple_mention_scorer']:
        event_mention_extractor = SimpleMentionExtractor(config, bert_model_hidden_size, device).to(device)
        event_mention_extractor.load_state_dict(torch.load(args.event_mention_extractor_path))

    else:
        event_mention_extractor = MentionExtractor(config, bert_model_hidden_size, device).to(device)
        event_mention_extractor.load_state_dict(torch.load(args.big_mention_extractor_path))




    # entity_mention_extractor = SimpleMentionExtractor(config, bert_model_hidden_size, device).to(device)
    # entity_mention_extractor.load_state_dict(torch.load(args.entity_mention_extractor_path))
    # entity_mention_extractor.eval()

    pairwise_classifier = SimplePairWiseClassifier(config, bert_model_hidden_size).to(device)
    pairwise_optimizer = optim.Adam(pairwise_classifier.parameters(), lr=config['learning_rate'],
                                    weight_decay=config['weight_decay'])


    graph_pairwise = GraphPairwiseClassifier(config, bert_model_hidden_size).to(device)
    graph_optimizer = optim.Adam(graph_pairwise.parameters(), lr=config['learning_rate'],
                                 weight_decay=config['weight_decay'])

    criterion = nn.BCEWithLogitsLoss()

    logger.info('Number of parameters of mention extractor: {}'.format(count_parameters(event_mention_extractor)))
    logger.info('Number of parameters of pairwise scorer: {}'.format(count_parameters(pairwise_classifier)))


    training_set = sp_tokens[0]
    all_topics, topic_list_of_docs, topic_origin_tokens, topic_bert_tokens, topic_start_end_bert = training_set
    logger.info('Number of topics: {}'.format(len(all_topics)))
    max_dev = (0, None)


    for epoch in range(config['epochs']):
        logger.info('Epoch: {}'.format(epoch))
        pairwise_classifier.train()

        if config['update_mention_scorer']:
            event_mention_extractor.train()
        else:
            event_mention_extractor.eval()


        event_mention_extractor.to(device)
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


            if config['update_mention_scorer']:
                event_span_embeddings, event_span_scores = event_mention_extractor(topic_start_end_embeddings,
                                                                                   topic_continuous_embeddings,
                                                                                   topic_width)
            else:

                with torch.no_grad():
                    event_span_embeddings, event_span_scores = event_mention_extractor(topic_start_end_embeddings,
                                                                                       topic_continuous_embeddings,
                                                                                   topic_width)

                # entity_span_embeddings, entity_span_scores = entity_mention_extractor(topic_start_end_embeddings,
                #                                                                       topic_continuous_embeddings,
                #                                                                       topic_width)

            if use_gold_mentions:
                event_span_indices = train_event_labels.nonzero().squeeze(1)
                # entity_span_indices = train_entity_labels.nonzero().squeeze(1)

            else:
                event_span_scores, event_span_indices = torch.topk(event_span_scores.squeeze(1), int(0.3 * num_of_tokens), sorted=False)
                event_span_indices, _ = torch.sort(event_span_indices)
                # entity_span_scores, entity_span_indices = torch.topk(entity_span_scores.squeeze(1),
                #                                                      int(0.4 * num_of_tokens), sorted=False)
                # entity_span_indices, _ = torch.sort(entity_span_indices)


            event_span_embeddings = event_span_embeddings[event_span_indices]
            train_event_labels = train_event_labels[event_span_indices]
            # entity_span_embeddings = entity_span_embeddings[entity_span_indices]
            # train_entity_labels = train_entity_labels[entity_span_indices]



            event_adj, entit_adj = create_adj_matrix(span_meta_data, event_span_indices, entity_span_indices, device)




            # a = graph_pairwise(event_span_embeddings, event_adj)





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




        logger.info('Number of positive pairs: {}'.format(positive_pairs))
        logger.info('Accumulate loss: {}'.format(accumulate_loss))
        logger.info('Evaluate on the dev set')


        dev_all_topics, dev_topic_list_of_docs, dev_topic_origin_tokens, dev_topic_bert_tokens, dev_topic_start_end_bert = sp_tokens[1]
        all_scores, all_labels = [], []


        event_mention_extractor.eval()
        pairwise_classifier.eval()


        for t, topic in enumerate(dev_all_topics):
            logger.debug('Topic {}'.format(topic))
            list_of_docs = dev_topic_list_of_docs[t]
            docs_original_tokens = dev_topic_origin_tokens[t]
            bert_tokens = dev_topic_bert_tokens[t]
            docs_bert_start_end = dev_topic_start_end_bert[t]

            docs_embeddings, docs_length = pad_and_read_bert(bert_tokens, bert_model, device)
            span_meta_data, span_embeddings, num_of_tokens = get_all_candidate_from_topic(
                config, device, list_of_docs, docs_original_tokens, docs_bert_start_end, docs_embeddings,
                docs_length, is_training)
            topic_start_end_embeddings, topic_continuous_embeddings, topic_width = span_embeddings
            torch.cuda.empty_cache()

            doc_id, sentence_id, start, end = span_meta_data
            dev_event_labels, dev_entity_labels = get_candidate_labels(device, doc_id, start, end, event_labels,
                                                                           entity_labels)



            with torch.no_grad():
                event_span_embeddings, event_span_scores = event_mention_extractor(topic_start_end_embeddings,
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
                pairwise_scores = pairwise_classifier(first_embeddings, second_embeddings)

            all_scores.extend(pairwise_scores.squeeze(1))
            all_labels.extend(pairwise_labels.to(torch.int))



        all_labels = torch.stack(all_labels)
        all_scores = torch.stack(all_scores)





        strict_preds = (all_scores > 0).to(torch.int)
        eval = Evaluation(strict_preds, all_labels)
        logger.info('Number of predictions: {}/{}'.format(strict_preds.sum(), len(strict_preds)))
        logger.info('Number of positive pairs: {}/{}'.format(len(all_labels.nonzero()), len(all_labels)))
        logger.info('Strict - Recall: {}, Precision: {}, F1: {}'.format(eval.get_recall(), eval.get_precision(), eval.get_f1()))

        # scores, indices = torch.topk(all_scores, int(0.05 * len(all_scores)), sorted=False)
        # rank_preds = torch.zeros(len(all_scores), device=device)
        # rank_preds[indices] = 1
        # eval = Evaluation(rank_preds, all_labels)
        # logger.info('Rank - Recall: {}, Precision: {}, F1: {}'.format(eval.get_recall(), eval.get_precision(), eval.get_f1()))

        if eval.get_f1() > max_dev[0]:
            max_dev = (eval.get_f1(), epoch)

            torch.save(pairwise_classifier.state_dict(), 'models/pairwise_gold_mentions')
            #
            # if config['update_mention_scorer']:
            #     torch.save(event_mention_extractor.state_dict(), 'models/2020_03_13/event_mention_extractor_5')

            # torch.save(pairwise_classifier.state_dict(), args.pairwise_path)


    logger.info('Best Performance: {}'.format(max_dev))


    user = 'gpus.experiment@gmail.com'
    pwd = 'Bension24'
    recipient = 'arie.cattan@gmail.com'
    subject = 'GPUs experiments are done'
    message = 'Best Dev F1: {}'.format(max_dev)

    # send_email(user, pwd, recipient, subject, message)