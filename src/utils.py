import numpy as np
import argparse
import xml.etree.ElementTree as ET
import os, fnmatch
from collections import OrderedDict

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='ECB+_LREC2014/ECB+', help='ECB+ data')
args = parser.parse_known_args()[0]



def arr_to_text(sentence_array):
    sentences = []
    for i, sentence in enumerate(sentence_array):
        sentence = str(i) + ' ' + ' '.join(list(map(lambda x: x[1], sentence)))
        sentences.append(sentence)
    return sentences

def get_source_mention(root, m_id):
    """
    get all the terms of a mention
    :param root: root of the document
    :param m_id: mention id
    :return: tuple (sentence id, mention terms)
    """
    terms_id = []

    for child in root[-2]: # Markables
        if child.attrib['m_id'] == str(m_id):
            for term in child:
                terms_id.append(term.attrib['t_id'])
            break

    terms_id = list(map(lambda x: int(x) - 1, terms_id))
    mention = 'sentence: ' + root[terms_id[0]].attrib['sentence'], ' '.join(list(map(lambda x: root[x].text, terms_id)))

    return mention

def get_target_mention(root, m_id):
    """
    get cluster description
    :param root: doc root
    :param m_id: mention id
    :return:
    """
    mention = ''
    for child in root[-2]:  # Markables
        if child.attrib['m_id'] == str(m_id):
            mention = child.attrib['TAG_DESCRIPTOR']
            break

    return mention

def print_relations(file_path, only_intra=True):
    tree = ET.parse(file_path)
    root = tree.getroot()
    for child in root[-1]:
        if not only_intra or child.tag == 'INTRA_DOC_COREF' or (child.tag == 'CROSS_DOC_COREF' and len(list(child)) > 2):
            print(child.tag, child.attrib)
            for mention in child:
                if mention.tag == 'source':
                    print(mention.tag, mention.attrib, get_source_mention(root, mention.attrib['m_id']))
                elif mention.tag == 'target':
                    print(mention.tag, mention.attrib, get_target_mention(root, mention.attrib['m_id']))



def get_mention_cluster(root, id_cluster):
    mentions = []
    for child in root[-1]: #Relations
        if 'note' in child.attrib and child.attrib['note'] == id_cluster:
            for mention in child:
                if mention.tag == 'source':
                    mentions.append(get_source_mention(root, mention.attrib['m_id']))

    return mentions



def get_clusters(folder_path):
    pattern = "*.xml"
    events = {}
    entities = {}

    for file in os.listdir(folder_path):
        if fnmatch.fnmatch(file, pattern):
            file_path = folder_path + '/' + file
            tree = ET.parse(file_path)
            root = tree.getroot()
            for child in root[-2]: #Markables
                if 'instance_id' in child.attrib:
                    id_cluster = child.attrib['instance_id']
                    mentions = get_mention_cluster(root, id_cluster)
                    if id_cluster.startswith('ACT') or id_cluster.startswith('NEG'):
                        events[id_cluster] = [] if id_cluster not in events else events[id_cluster]
                        events[id_cluster].append([file, mentions])
                    else:
                        entities[id_cluster] = [] if id_cluster not in entities else entities[id_cluster]
                        entities[id_cluster].append([file, mentions])


    return events, entities

def get_clusters_by_subtopic(folder_path):
    pattern = "*.xml"
    events = {}
    entities = {}
    events['ECB'] = {}
    events['ECB+'] = {}
    entities['ECB'] = {}
    entities['ECB+'] = {}


    for file in os.listdir(folder_path):
        if fnmatch.fnmatch(file, pattern):
            file_path = folder_path + '/' + file
            tree = ET.parse(file_path)
            root = tree.getroot()
            subtopic = 'ECB' if file_path.endswith('ecb.xml') else 'ECB+'

            for child in root[-2]: #Markables
                if 'instance_id' in child.attrib:
                    id_cluster = child.attrib['instance_id']
                    mentions = get_mention_cluster(root, id_cluster)
                    if id_cluster.startswith('ACT') or id_cluster.startswith('NEG'):
                        events[subtopic][id_cluster] = [] if id_cluster not in events[subtopic] else events[subtopic][id_cluster]
                        events[subtopic][id_cluster].append([file, mentions])
                    else:
                        entities[subtopic][id_cluster] = [] if id_cluster not in entities[subtopic] else entities[subtopic][id_cluster]
                        entities[subtopic][id_cluster].append([file, mentions])


    return events, entities



def get_metainfo(clusters):
    """
    print num of mentions per clusters
    :param clusters:
    :return:
    """
    dic = {}
    for cluster, doc_mention in clusters.items():
        num_of_mentions = sum([len(mention[1]) for mention in doc_mention])
        dic[num_of_mentions] = dic.get(num_of_mentions, 0) + 1

    for length, num_of_clusters in sorted(dic.items()):
        print("There are {} clusters with {} mentions".format(num_of_clusters, length))


def get_cluster_by_mention_num(clusters, num):
    clusters_names = []
    for cluster, doc_mention in clusters.items():
        num_of_mentions = sum([len(mention[1]) for mention in doc_mention])
        if num_of_mentions == num:
            clusters_names.append(cluster)

    return clusters_names


def get_all_mentions_from_cluster(clusters):
    mentions = {}
    for cluster, doc_mention in clusters.items():
        for doc, sentence_mention in doc_mention:
            for sentence_num, mention in sentence_mention:
                mentions[mention] = [] if mention not in mentions else mentions[mention]
                mentions[mention].append([cluster, sentence_num, doc])


    return mentions



if __name__ == '__main__':
    os.chdir('ECB+_LREC2014/ECB+')
    file_path = '2/2_1ecbplus.xml'

    folder_path = '35'

    events, entities = get_clusters(folder_path)
    a, b = get_clusters_by_subtopic(folder_path)

    event_mentions = get_all_mentions_from_cluster(events)
    entity_mentions = get_all_mentions_from_cluster(entities)


    #print('Number of mentions of the cluster: {}'.format(entity_mentions['ACT17642063946729293']))
