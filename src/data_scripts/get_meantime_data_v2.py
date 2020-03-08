# -*- coding: utf-8 -*-
import numpy as np
import xml.etree.ElementTree as ET
import os, fnmatch
import argparse
import time
import json
import spacy
from utils import *


TRAIN = ['corpus_airbus', 'corpus_apple']
VALIDATION = ['corpus_gm']
TEST = ['corpus_stock']



event_singleton_idx, entity_singleton_idx = int(1E8), int(2E8)

def obj_dict(obj):
    return obj.__dict__


def align_ecb_bert_tokens(ecb_tokens, bert_tokens):
    bert_to_ecb_ids = []
    relative_char_pointer = 0
    ecb_token = None
    ecb_token_id = None

    for bert_token in bert_tokens:
        if relative_char_pointer == 0:
            ecb_token_id, ecb_token = ecb_tokens.pop(0)

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



def get_mention_doc(root, doc_name):
    entity_mentions, event_mentions = [], []
    mentions_fields, mention_cluster_info = {}, {}
    relation_source_target, relation_rid = {}, {}

    for mention in root.find('Markables'):
        m_id = mention.attrib['m_id']

        if mention.tag == 'ENTITY_MENTION' or mention.tag == 'EVENT_MENTION':
            event = True if mention.tag == 'EVENT_MENTION' else False
            tokens_ids = [int(term.attrib['t_id'])  for term in mention]
            if not tokens_ids:
                continue
            sentence = root[tokens_ids[0] - 1].attrib['sentence']
            if int(sentence) >= 6:
                continue
            tokens = ' '.join(list(map(lambda x: root[x-1].text, tokens_ids)))
            lemmas, tags = [], []
            for tok in nlp(tokens):
                lemmas.append(tok.lemma_)
                tags.append(tok.tag_)


            mentions_fields[m_id] = {
                "doc_id": doc_name,
                "m_id": m_id,
                "sentence_id" : sentence,
                "tokens_ids": tokens_ids,
                "tokens": tokens,
                "tags": ' '.join(tags),
                "lemmas": ' '.join(lemmas),
                "event": event
            }


        elif mention.tag == 'ENTITY' or mention.tag == 'EVENT':
            mention_cluster_info[m_id] = {
                "cluster_id": mention.attrib.get('instance_id', ''),
                "cluster_desc": mention.attrib.get('TAG_DESCRIPTOR', '')
            }

    for relation in root.find('Relations'):
        if relation.tag == 'REFERS_TO':
            target_mention = relation[-1].attrib['m_id']
            relation_rid[target_mention] = relation.attrib['r_id']
            for mention in relation:
                if mention.tag == 'source':
                    relation_source_target[mention.attrib['m_id']] = target_mention

    global event_singleton_idx, entity_singleton_idx

    for m_id, mention in mentions_fields.items():
        target = relation_source_target.get(m_id, None)
        if target is None or mention_cluster_info[target]['cluster_id'] == '':
            if mention['event']:
                cluster_id = event_singleton_idx
                event_singleton_idx += 1
            else:
                cluster_id = entity_singleton_idx
                entity_singleton_idx += 1
        else:
            cluster_id = int(mention_cluster_info[target]['cluster_id'][3:])

        cluster_desc = '' if target is None else mention_cluster_info[target]['cluster_desc']


        mention_info = mention.copy()
        mention_info["cluster_id"] = cluster_id
        mention_info["cluster_desc"] = cluster_desc
        event = mention_info.pop("event")
        if event:
            event_mentions.append(mention_info)
        else:
            entity_mentions.append(mention_info)


    return event_mentions, entity_mentions


def get_clusters(mentions):
    clusters = {}
    for mention in mentions:
        cluster_id = mention['cluster_id']
        clusters[cluster_id] = [] if cluster_id not in clusters else clusters[cluster_id]
        clusters[cluster_id].append(mention)

    return clusters


def read_topic(topic_path):
    all_docs = {}
    pattern = '*xml'
    all_event_mentions, all_entity_mentions = [], []
    topic = topic_path.split('/')[-1]

    for doc in os.listdir(topic_path):
        if fnmatch.fnmatch(doc, pattern):
            doc_path = os.path.join(topic_path, doc)
            tree = ET.parse(doc_path)
            root = tree.getroot()
            selected_sentences = list(range(6)) #only the fist 5 sentences are considered

            # Extract all the event and entity mentions
            event_mentions, entity_mentions = get_mention_doc(root, doc)
            all_event_mentions += event_mentions
            all_entity_mentions += entity_mentions

            original_tokens = []
            for child in root:
                if child.tag == 'token' and int(child.attrib['sentence']) in selected_sentences:
                    original_tokens.append(
                        [int(child.attrib['sentence']), int(child.attrib['t_id']), child.text.replace('�', '').strip()])

            all_docs[doc] = original_tokens


    event_clusters = get_clusters(all_event_mentions)
    entity_clusters = get_clusters(all_entity_mentions)
    event_singleton_cluster_flag = {c: True if len(m) == 1 else False for c, m in event_clusters.items()}
    entity_singleton_cluster_flag = {c: True if len(m) == 1 else False for c, m in entity_clusters.items()}
    for item in all_event_mentions:
        item.update({'topic': topic, 'singleton': event_singleton_cluster_flag[item['cluster_id']]})
    for item in all_entity_mentions:
        item.update({'topic': topic, 'singleton': entity_singleton_cluster_flag[item['cluster_id']]})

    return all_docs, all_event_mentions, all_entity_mentions


def get_all_docs(data_path):
    train_docs, train_event_mentions, train_entity_mentions = {}, [], []
    dev_docs, dev_event_mentions, dev_entity_mentions = {}, [], []
    test_docs, test_event_mentions, test_entity_mentions = {}, [], []
    for topic in os.listdir(data_path):
        topic_path = os.path.join(data_path, topic)
        if os.path.isdir(topic_path):
            print('Processing topic {}'.format(topic))
            topic_docs, event_mentions, entity_mentions = read_topic(topic_path)

            if topic in TRAIN:
                train_docs.update(topic_docs)
                train_event_mentions += event_mentions
                train_entity_mentions += entity_mentions

            elif topic in VALIDATION:
                dev_docs.update(topic_docs)
                dev_event_mentions += event_mentions
                dev_entity_mentions += entity_mentions

            else:
                test_docs.update(topic_docs)
                test_event_mentions += event_mentions
                test_entity_mentions += entity_mentions


    return (train_docs, train_event_mentions, train_entity_mentions), \
           (dev_docs, dev_event_mentions, dev_entity_mentions),\
           (test_docs, test_event_mentions, test_entity_mentions)


def get_stats(entity_mentions, event_mentions):
    entity_clusters = get_clusters(entity_mentions)
    event_clusters = get_clusters(event_mentions)
    print('Event clusters: {}'.format(len(event_clusters)))
    print('Event mentions: {}'.format(len(event_mentions)))
    print('Event singletons mentions: {}'.format(
        sum([1 for l in event_mentions if l['singleton']])))
    print('Entity clusters: {}'.format(len(entity_clusters)))
    print('Entity mentions: {}'.format(len(entity_mentions)))
    print('Entity singletons mentions: {}'.format(
        sum([1 for l in entity_mentions if l['singleton']])))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parsing MEANTIME corpus')
    parser.add_argument('--data_path', type=str,
                        default='data/datasets/meantime_newsreader_english_oct15/intra_cross-doc_annotation',
                        help=' Path to ECB+ corpus')
    parser.add_argument('--output_dir', type=str, default='data/meantime/mentions',
                        help=' The directory of the output files')
    args = parser.parse_args()

    nlp = spacy.load('en_core_web_sm', disable=['textcat'])

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print('Getting all mentions')
    train, dev, test = get_all_docs(args.data_path)
    docs = train[0], dev[0], test[0]
    event_mentions = train[1], dev[1], test[1]
    entity_mentions = train[2], dev[2], test[2]


    for i, type in enumerate(['train', 'dev', 'test']):
        print('Statistics on {}'.format(type))
        get_stats(entity_mentions[i], event_mentions[i])

        with open(os.path.join(args.output_dir, type + '.json'), 'w') as f:
            json.dump(docs[i], f, indent=4)
        with open(os.path.join(args.output_dir, type + '_events.json'), 'w') as f:
            json.dump(event_mentions[i], f, default=obj_dict, indent=4, ensure_ascii=False)
        with open(os.path.join(args.output_dir, type + '_entities.json'), 'w') as f:
            json.dump(entity_mentions[i], f, default=obj_dict, indent=4, ensure_ascii=False)

