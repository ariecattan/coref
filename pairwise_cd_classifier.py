import pandas as pd
import torch
import argparse
import json
import pickle
import os
from itertools import combinations
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from random import sample
import torch.optim as optim
from sklearn.utils import shuffle
import logging
import pyhocon
import random
from datetime import datetime
from classifier import PairWiseClassifier
from evaluator import Evaluation



def get_span_embedding(mentions, bert_embeddings):
    mention_spans = []
    for index, mention in mentions.iterrows():
        doc_id, bert_start_index, bert_end_index = mention['doc_id'], mention['bert_start_index'], mention['bert_end_index']
        doc_embeddings = bert_embeddings[doc_id]
        if bert_end_index - bert_start_index < 1:
            raise ValueError(doc_id, bert_start_index, bert_end_index)
        mention_span = doc_embeddings[torch.tensor(range(bert_start_index, bert_end_index), dtype=torch.long)].sum(0)
        mention_spans.append(mention_span)

    return torch.stack(mention_spans)




def prepare_examples(mentions):
    positives, negatives = [], []
    positive_same_lemma, negative_same_lemma = [], []
    grouped_by_topic = mentions.groupby('topic')
    for topic, grouped_mentions_by_topic in grouped_by_topic:
        index_pair = list(combinations(grouped_mentions_by_topic['index'].to_numpy(), 2))
        grouped_by_cluster = grouped_mentions_by_topic.groupby('cluster_id')
        positive_pairs = []
        for cluster, grouped_mentions_by_cluster in grouped_by_cluster:
            positive_pairs.extend(list(combinations(grouped_mentions_by_cluster['index'].to_numpy(), 2)))
        negative_pairs = set(index_pair) - set(positive_pairs)

        grouped_by_lemmas = grouped_mentions_by_topic.groupby('lemmas')
        same_lemma = []
        for lemma, grouped_mention_by_lemmas in grouped_by_lemmas:
            same_lemma.extend(list(combinations(grouped_mention_by_lemmas['index'], 2)))

        positive_same_lemma_pairs = set(positive_pairs).intersection(set(same_lemma))
        negative_same_lemma_pairs = negative_pairs.intersection(same_lemma)


        negatives.extend(list(negative_pairs))
        positives.extend(positive_pairs)
        positive_same_lemma.extend(list(positive_same_lemma_pairs))
        negative_same_lemma.extend(list(negative_same_lemma_pairs))


    return (positives, negatives), (positive_same_lemma, negative_same_lemma)




def tensorize_instances(positives, negatives, negative_balance):
    if negative_balance and negative_balance * len(positives) < len(negatives):
        negatives = sample(negatives, negative_balance * len(positives))
    labels = torch.cat((torch.tensor([1]).repeat(len(positives)), torch.tensor([0]).repeat(len(negatives))))
    first_mentions, second_mentions = zip(*(positives + negatives))
    return torch.tensor(first_mentions, dtype=torch.long), torch.tensor(second_mentions, dtype=torch.long), labels


def get_pairwise_candidate(mentions, same_lemma=False, negative_balance=None):
    regular_instances, same_lemma_instances = prepare_examples(mentions)
    if same_lemma:
        first_mentions, second_mentions, labels = tensorize_instances(regular_instances[0], same_lemma_instances[1], negative_balance)
    else:
        first_mentions, second_mentions, labels = tensorize_instances(regular_instances[0], regular_instances[1], negative_balance)

    return first_mentions, second_mentions, labels



def predict(model, first_mentions, second_mentions, embeddings, config):
    model.eval()
    pairwise_span = torch.cat((embeddings[first_mentions], embeddings[second_mentions]), dim=1)
    if config['use_element_wise_mult']:
        pairwise_span = torch.cat((pairwise_span, embeddings[first_mentions] * embeddings[second_mentions]), dim=1)

    with torch.no_grad():
        output = model(pairwise_span).squeeze(1)
        predictions = (output > 0).to(torch.int)

    return predictions



def evaluate_model(model, dataset, embeddings, config):
    model.eval()
    first_mentions, second_mentions, labels = dataset
    predictions = predict(model, first_mentions, second_mentions, embeddings, config)
    evaluation = Evaluation(predictions, labels.to(config['device']))
    return evaluation




def train(model, train_set, train_embeddings, dev_set, dev_embeddings, loss_fn, optimizer, config):
    first_mentions, second_mentions, labels = train_set
    logging.info('Training..')
    example_ids = list(range(len(labels)))
    running_loss, train_ev, dev_ev = [], [], []

    dev_evaluator = evaluate_model(model, dev_set, dev_embeddings, config)
    dev_ev.append(dev_evaluator)

    for i, epoch in enumerate(range(config['epochs'])):
        loss_per_epoch = 0.0
        model.train()
        logging.info('Epoch: {}'.format(epoch))
        example_ids = shuffle(example_ids)
        for i in range(0, len(labels), config['batch_size']):
            optimizer.zero_grad()
            indices = example_ids[i:i+config['batch_size']]
            idx1, idx2 = first_mentions[indices], second_mentions[indices]
            m1, m2 = train_embeddings[idx1], train_embeddings[idx2]
            pairwise_span = torch.cat((m1, m2), dim=1)
            if config['use_element_wise_mult']:
                pairwise_span = torch.cat((pairwise_span, m1 * m2), dim=1)
            output = model(pairwise_span)
            loss = loss_fn(output.squeeze(), labels[indices].to(device))
            loss.backward()
            optimizer.step()


            loss_per_epoch += loss.item()

        running_loss.append(loss_per_epoch / i)
        dev_evaluator = evaluate_model(model, dev_set, dev_embeddings, config)
        dev_ev.append(dev_evaluator)
        logging.info("Precision/Recall on the dev set: {}/{}".format(dev_evaluator.get_precision(), dev_evaluator.get_recall()))
        logging.info("F1 on the dev set: {}".format(dev_evaluator.get_f1()))


    return model, running_loss, dev_ev



def get_similariry(mention1, mention2):
    if mention1['tokens'] == mention2['tokens'] or mention1['lemmas'] == mention2['lemmas']:
        return True
    return False


def get_stat_positive_examples(dataset, dataframe):
    first_mentions, second_mentions, labels = dataset
    exact_string = 0
    same_lemma = 0


    for m1, m2 in zip(first_mentions, second_mentions):
        mention1, mention2 = dataframe.iloc[m1.item()], dataframe.iloc[m2.item()]
        if mention1['tokens'] == mention2['tokens']:
            exact_string += 1
        elif mention1['lemmas'] == mention2['lemmas']:
            same_lemma += 1

    return exact_string, same_lemma


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='experiments/config_file.json')
    args = parser.parse_args()
    config = pyhocon.ConfigFactory.parse_file(args.config)

    experiment_name = str(os.path.basename(args.config)).split('.')[0] + '_' + '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
    if not os.path.exists(os.path.join(config['save_path'], experiment_name)):
        os.makedirs(os.path.join(config['save_path'], experiment_name))

    format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
    logging.basicConfig(format=format, level=logging.INFO,
                        filename=os.path.join(config['save_path'], experiment_name, 'log.txt'),
                        filemode='w', datefmt='%Y-%m-%d %H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler())
    logger = logging.getLogger(__name__)
    logging.info(pyhocon.HOCONConverter.convert(config, "hocon"))

    device = torch.device('cuda:{}'.format(config['gpu']) if torch.cuda.is_available() and config['gpu'] > 0 else 'cpu')
    if device.type == 'cpu':
        logging.info('Using CPU')
    else:
        logging.info('Using GPU:{}'.format(config['gpu']))

    config['device'] = device


    torch.manual_seed(config['random_seed'])
    random.seed(config['random_seed'])
    np.random.seed(config['random_seed'])
    if device.type == 'gpu':
        torch.cuda.manual_seed(config['random_seed'])


    logging.info('Reading bert embeddings..')
    with open(os.path.join(config['bert_embeddings'], config['bert_model'] + '_' + str(config['bert_segment_length'])), 'rb') as f:
        bert_embeddings = pickle.load(f)

    logging.info('Reading and preparing the data..')
    train_event_mentions = pd.read_json(config['train_event_path']).reset_index()
    train_event_mention_embeddings = get_span_embedding(train_event_mentions, bert_embeddings)
    train_event_pairwise_candidate = get_pairwise_candidate(train_event_mentions, config["training_same_lemma"], config['negative_balance'])
    dev_event_mentions = pd.read_json(config['dev_event_path']).reset_index()
    dev_event_entions_embeddings = get_span_embedding(dev_event_mentions, bert_embeddings)
    dev_event_pairwise_candidate = get_pairwise_candidate(dev_event_mentions)

    event_pos_idx = train_event_pairwise_candidate[-1].nonzero().squeeze()
    first_event, second_event = train_event_pairwise_candidate[0][event_pos_idx], \
                                  train_event_pairwise_candidate[1][event_pos_idx]
    positive_examples, negative_examples = (train_event_pairwise_candidate[-1] == 1).float().sum().item(), (train_event_pairwise_candidate[-1] == 0).float().sum().item()
    logging.info('Positive / Negative data {} / {}'.format(positive_examples, negative_examples))

    
    
    model = PairWiseClassifier(config).to(config['device'])
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    model, running_loss, dev_ev = train(model, train_event_pairwise_candidate, train_event_mention_embeddings, dev_event_pairwise_candidate, dev_event_entions_embeddings, loss_fn, optimizer, config)

    accuracies = list(map(lambda x: x.get_accuracy(), dev_ev))
    precision_recall = list(map(lambda x: (x.get_precision(), x.get_recall()), dev_ev))
    f1 = list(map(lambda x: x.get_f1(), dev_ev))


    os.chdir(os.path.join(config['save_path'], experiment_name))
    np.save('loss', running_loss)
    np.save('dev accuracy', accuracies)
    np.save('precision_recall', precision_recall)
    np.save('f1', f1)

    logging.info('Maximum F1 score on the dev set: {}'.format(max(f1)))
    
