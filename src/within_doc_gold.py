import argparse
import json
import spacy
import numpy as np
import os
import codecs
from search_data import *
import operator

parser = argparse.ArgumentParser(description='Creating Within Document Representation')

parser.add_argument('--data', type=str, default='data', help='Path to dataset')


parser.add_argument('--embedding_path', type=str, default='glove_ecb+',
                        help='Path to embedding folder')

args = parser.parse_args()
nlp = spacy.load('en_core_web_sm')


def load_embeddings(embeddings_folder_path):
    vocab_file = ''
    embedding_file = ''

    for file in os.listdir(embeddings_folder_path):
        if file.endswith('vocab'):
            vocab_file = embeddings_folder_path + '/' + file
        elif file.endswith('npy'):
            embedding_file = embeddings_folder_path + '/' + file

    if vocab_file == '' or embedding_file == '':
        return('Embedding or vocab file does not exist in the folder')

    with codecs.open(vocab_file, 'r', 'utf-8') as f:
        vocab = [line.strip() for line in f]

    w2i = {w: i for i, w in enumerate(vocab)}
    wv = np.load(embedding_file)

    return wv, np.array(vocab), w2i


def get_token_embedding(token, wv, w2i):
    if token in w2i:
        return wv[w2i[token]]
    print('Unknown word')
    return wv[w2i['unk']]



def get_cluster_representation(within_doc_cluster, wv, w2i):
    cluster = np.zeros(wv.shape[1] * 2)
    mentions = {}
    for mention in within_doc_cluster:
        mention_text = mention['MENTION_TEXT']
        doc = nlp(mention_text)
        flag = True
        for token in doc:
            if token.tag_ == 'PRP' or token.tag_ == 'WP':
                flag = False
        if flag:
            mentions[mention_text] = mentions.get(mention_text, 0) + 1


    most_popular_mention = max(mentions.items(), key=operator.itemgetter(1))[0]
    tokens = most_popular_mention.split(' ')
    vec = sum([get_token_embedding(token, wv, w2i) for token in tokens]) / len(tokens)
    cluster[:wv.shape[1]] = vec
    del mentions[most_popular_mention]

    for mention in mentions:
        tokens = mention.split(' ')
        mention_rep = np.zeros(wv.shape[1])
        for token in tokens:
            mention_rep += get_token_embedding(token, wv, w2i)

        mention_rep /= len(tokens)

        cluster[wv.shape[1]:] += mention_rep


    return cluster

if __name__ == '__main__':

    wv, vocab, w2i = load_embeddings(args.embedding_path)
    data = args.data

    with open(data + '/train.json', 'r') as f:
        train_data = json.load(f)

    event_within_doc, entity_within_doc = get_gold_within_doc(train_data)


    a = get_cluster_representation(entity_within_doc['HUM16236184328979740_1_14ecb.xml'], wv, w2i)
    b = get_cluster_representation(entity_within_doc['HUM16195623414690294_4_4ecbplus.xml'], wv, w2i)
    c = get_cluster_representation(entity_within_doc['HUM16195623414690294_4_10ecbplus.xml'], wv, w2i)


