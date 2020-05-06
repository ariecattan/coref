import collections
import logging
import os
import torch
import random
import numpy as np
import smtplib
import torch.optim as optim
import json
from datetime import datetime
import pickle

from corpus import Corpus



def create_corpus(config, tokenizer, split_name, use_gold_mentions=True):
    docs_path = os.path.join(config['data_folder'], split_name + '.json')
    mentions_path = os.path.join(config['data_folder'],
                                 split_name + '_{}.json'.format(config['mention_type']))
    with open(docs_path, 'r') as f:
        documents = json.load(f)

    mentions = []
    if use_gold_mentions:
        with open(mentions_path, 'r') as f:
            mentions = json.load(f)

    predicted_topics = None
    if config['use_predicted_subtopics']:
        predicted_topics_path = '/home/nlp/ariecattan/event_entity_coref_ecb_plus/data/external/document_clustering/predicted_topics'
        with open(predicted_topics_path, 'rb') as f:
            predicted_topics = pickle.load(f)

    corpus = Corpus(documents, tokenizer, mentions, predicted_topics)

    return corpus


def create_logger(config, create_file=True):
    logging.basicConfig(datefmt='%Y-%m-%d %H:%M:%S', format='w')
    logger = logging.getLogger(__name__)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging.DEBUG)
    c_handler.setFormatter(formatter)
    logger.addHandler(c_handler)

    if create_file:
        if not os.path.exists(config['log_path']):
            os.makedirs(config['log_path'])

        f_handler = logging.FileHandler(
            os.path.join(config['log_path'],'{}.txt'.format(datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))), mode='w')
        f_handler.setLevel(logging.INFO)
        f_handler.setFormatter(formatter)
        logger.addHandler(f_handler)
    logger.propagate = False

    return logger


def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def fix_seed(config):
    torch.manual_seed(config['random_seed'])
    random.seed(config['random_seed'])
    np.random.seed(config['random_seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['random_seed'])
        torch.cuda.manual_seed_all(config['random_seed'])


def get_loss_function(config):
    if config['loss'] == 'hinge':
        return torch.nn.HingeEmbeddingLoss()
    else:
        return torch.nn.BCEWithLogitsLoss()


def get_optimizer(config, models):
    parameters = []
    for model in models:
        parameters += list(model.parameters())

    if config['optimizer'] == "adam":
        return optim.Adam(parameters, lr=config['learning_rate'], weight_decay=config['weight_decay'])
    else:
        return optim.SGD(parameters, lr=config['learning_rate'], weight_decay=config['weight_decay'])


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)






def add_to_dic(dic, key, val):
    if key not in dic:
        dic[key] = []
    dic[key].append(val)


def send_email(user, pwd, recipient, subject, body):

    FROM = user
    TO = recipient if isinstance(recipient, list) else [recipient]
    SUBJECT = subject
    TEXT = body

    # Prepare actual message
    message = """From: %s\nTo: %s\nSubject: %s\n\n%s
    """ % (FROM, ", ".join(TO), SUBJECT, TEXT)
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.ehlo()
        server.starttls()
        server.login(user, pwd)
        server.sendmail(FROM, TO, message)
        server.close()
        print('successfully sent the mail')
    except:
        print("failed to send mail")





def align_ecb_bert_tokens(ecb_tokens, bert_tokens):
    bert_to_ecb_ids = []
    relative_char_pointer = 0
    ecb_token = None
    ecb_token_id = None

    for bert_token in bert_tokens:
        if relative_char_pointer == 0:
            ecb_token_id, ecb_token, _, _ = ecb_tokens.pop(0)

        bert_token = bert_token.replace("##", "")
        if bert_token == ecb_token:
            bert_to_ecb_ids.append(ecb_token_id)
            relative_char_pointer = 0
        elif ecb_token.find(bert_token) == 0:
            bert_to_ecb_ids.append(ecb_token_id)
            relative_char_pointer = len(bert_token)
            ecb_token = ecb_token[relative_char_pointer:]
        else:
            print("When bert token is longer?")
            raise ValueError((bert_token, ecb_token))

    return bert_to_ecb_ids