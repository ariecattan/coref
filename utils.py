import collections
import logging
import os
from itertools import chain
import torch
import random
import numpy as np

def flatten(d, parent_key='', sep=''):
    items = []
    for k, v in d.items():
        new_key = k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)



def obj_dict(obj):
    return obj.__dict__


def get_list_annotated_sentences(annotated_sentences):
    sentences = {}
    for topic, doc, sentence in annotated_sentences:
        if topic not in sentences:
            sentences[topic] = {}
        doc_name = topic + '_' + doc + '.xml'
        if doc_name not in sentences[topic]:
            sentences[topic][doc_name] = []
        sentences[topic][doc_name].append(sentence)
    return sentences


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



def create_logger(config):
    logging.basicConfig(datefmt='%Y-%m-%d %H:%M:%S', format='w')
    logger = logging.getLogger(__name__)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging.DEBUG)
    c_handler.setFormatter(formatter)
    f_handler = logging.FileHandler(os.path.join(config['save_path'], 'log{}.txt'.format(config['exp_num'])), mode='w')
    f_handler.setLevel(logging.INFO)
    f_handler.setFormatter(formatter)
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
    logger.propagate = False

    return logger


def separate_docs_into_topics(texts):
    text_by_topics = {}
    for k, v in texts.items():
        topic_key = k.split('_')[0]
        if topic_key not in text_by_topics:
            text_by_topics[topic_key] = {}
        text_by_topics[topic_key][k] = v

    return text_by_topics


def get_mentions_by_doc(mentions):
    mentions_by_doc = {}
    for m in list(chain.from_iterable(mentions)):
        if m['doc_id'] not in mentions_by_doc:
            mentions_by_doc[m['doc_id']] = []
        mentions_by_doc[m['doc_id']].append(m)

    return mentions_by_doc


def get_dict_labels(mentions):
    label_dict = {}
    for m in list(chain.from_iterable(mentions)):
        if m['doc_id'] not in label_dict:
            label_dict[m['doc_id']] = {}
        label_dict[m['doc_id']][(min(m['tokens_ids']), max(m['tokens_ids']))] = m['cluster_id']

    return label_dict


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fix_seed(config):
    torch.manual_seed(config['random_seed'])
    random.seed(config['random_seed'])
    np.random.seed(config['random_seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['random_seed'])


def split_doc_into_segments(token_ids, sentence_ids, segment_length=512):
    segments = [0]
    current_token = 0
    while current_token < len(token_ids):
        end_token = min(len(token_ids) - 1, current_token + segment_length - 1)
        sentence_end = sentence_ids[end_token]
        if end_token != len(token_ids) - 1 and sentence_ids[end_token + 1] == sentence_end:
            while end_token >= current_token and sentence_ids[end_token] == sentence_end:
                end_token -= 1

            if end_token < current_token:
                raise ValueError(token_ids)

        current_token = end_token + 1
        segments.append(current_token)

    return segments