import collections
import logging
import os
from itertools import chain
import torch
import random
import numpy as np
from itertools import compress, combinations
import smtplib
import torch.optim as optim



def create_logger(config, create_file=True):
    logging.basicConfig(datefmt='%Y-%m-%d %H:%M:%S', format='w')
    logger = logging.getLogger(__name__)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging.DEBUG)
    c_handler.setFormatter(formatter)
    logger.addHandler(c_handler)

    if create_file:
        f_handler = logging.FileHandler(os.path.join(config['save_path'], 'log{}.txt'.format(config['exp_num'])), mode='w')
        f_handler.setLevel(logging.INFO)
        f_handler.setFormatter(formatter)
        logger.addHandler(f_handler)
    logger.propagate = False

    return logger


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



def get_mentions_by_doc(mentions):
    mentions_by_doc = {}
    for m in list(chain.from_iterable(mentions)):
        if m['doc_id'] not in mentions_by_doc:
            mentions_by_doc[m['doc_id']] = []
        mentions_by_doc[m['doc_id']].append(m)

    return mentions_by_doc


def get_dict_labels(mentions):
    if len(mentions) in  (2, 3):
        mentions = list(chain.from_iterable(mentions))
    label_dict = {}
    for m in mentions:
        if m['doc_id'] not in label_dict:
            label_dict[m['doc_id']] = {}
        label_dict[m['doc_id']][(min(m['tokens_ids']), max(m['tokens_ids']))] = m['cluster_id']

    return label_dict


def separate_docs_into_topics(texts, subtopic=True):
    text_by_topics = {}
    for k, v in texts.items():
        topic_key = k.split('_')[0]
        if subtopic:
            topic_key += '_{}'.format(1 if 'plus' in k else 0)
        if topic_key not in text_by_topics:
            text_by_topics[topic_key] = {}
        text_by_topics[topic_key][k] = v

    return text_by_topics




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






def pad_tokens_for_attention(continuous_embeddings):
    device = continuous_embeddings[0].device
    zero_vec = torch.zeros(1024, device=device)
    lengths = [len(x) for x in continuous_embeddings]
    max_length = max(lengths)

    padded_tokens_embeddings = []
    masks = torch.zeros((len(continuous_embeddings), max_length), device=device)
    for i, mention in enumerate(continuous_embeddings):
        length = lengths[i]
        padded_tokens_embeddings.append(
            torch.cat((mention, zero_vec.repeat(max_length - length, 1))))
        masks[i][list(range(length))] = 1

    return torch.stack(padded_tokens_embeddings), masks


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
