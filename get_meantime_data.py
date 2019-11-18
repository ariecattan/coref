# -*- coding: utf-8 -*-
import numpy as np
import xml.etree.ElementTree as ET
import os, fnmatch
import argparse
import time
import json
import spacy


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



TRAIN = ['corpus_airbus', 'corpus_apple']
VALIDATION = ['corpus_gm']
TEST = ['corpus_stock']


parser = argparse.ArgumentParser(description='Parsing MEANTIME corpus')


parser.add_argument('--data_path', type=str, default='data/datasets/meantime_newsreader_english_oct15/intra_cross-doc_annotation',
                    help=' Path to the corpus')

parser.add_argument('--output_dir', type=str, default='data/meantime/mentions',
                        help=' The directory of the output files')

parser.add_argument('--with_pos', type=str2bool, default=False, help='Boolean value to include the pos tag of the mentions')

parser.add_argument('--entire_doc', type=str2bool, default=False, help='To load whole documents or only the first 5 sentences '
                                                                       'where the sentence num1 is not considered a sentence because it contains only the date of the article')
args = parser.parse_args()

if args.with_pos:
    nlp = spacy.load('en_core_web_sm')


def obj_dict(obj):
    return obj.__dict__


def get_all_mention(corpus_path, output_dir):
    train_events = []
    train_entities = []
    dev_events = []
    dev_entities = []
    test_events = []
    test_entities = []
    vocab = set()

    train_sentences = []
    dev_sentences = []
    test_sentences = []

    for folder in os.listdir(corpus_path):
        folder_path = corpus_path + '/' + folder
        if os.path.isdir(folder_path):
            event_mentions, entity_mentions, files, voc = get_topic_mention(folder_path)
            vocab.update(voc)
            if folder in TRAIN:
                train_events.extend(event_mentions)
                train_entities.extend(entity_mentions)
                train_sentences.extend(files)
            elif folder in VALIDATION:
                dev_events.extend(event_mentions)
                dev_entities.extend(entity_mentions)
                dev_sentences.extend(files)
            elif folder in TEST:
                test_events.extend(event_mentions)
                test_entities.extend(entity_mentions)
                test_sentences.extend(files)

    all_events = train_events + dev_events + test_events
    all_entities = train_entities + dev_entities + test_entities
    all_sentences = train_sentences + dev_sentences + test_sentences

    save_json(train_events, output_dir + '/train_event_gold_mentions.json')
    save_json(dev_events, output_dir + '/dev_event_gold_mentions.json')
    save_json(test_events, output_dir + '/test_event_gold_mentions.json')
    save_json(train_entities, output_dir + '/train_entity_gold_mentions.json')
    save_json(dev_entities, output_dir + '/dev_entity_gold_mentions.json')
    save_json(test_entities, output_dir + '/test_entity_gold_mentions.json')
    save_json(all_events, output_dir + '/all_event_gold_mentions.json')
    save_json(all_entities, output_dir + '/all_entity_gold_mentions.json')

    save_txt(train_sentences, output_dir + '/train_text.txt')
    save_txt(dev_sentences, output_dir + '/dev_text.txt')
    save_txt(test_sentences, output_dir + '/test_text.txt')
    save_txt(all_sentences, output_dir + '/all_text.txt')

    with open(output_dir + '/vocab', 'w') as f:
        for word in vocab:
            f.write(word + '\n')


    return (train_events, train_entities),\
           (dev_events, dev_entities), \
           (test_events, test_entities), \
           (all_events, all_entities), \
           vocab



def save_json(dic, file_name):
    with open(file_name, 'w') as f:
        json.dump(dic, f, default=obj_dict, indent=4, sort_keys=True)


def save_txt(data, file_name):
    with open(file_name, 'w') as f:
        for item in data:
            f.write("%s\n" % '\t'.join(item))


def get_topic_mention(topic_path):
    event_mentions = []
    entity_mentions = []
    pattern = '*.xml'
    vocab = set()
    files = []

    for file in os.listdir(topic_path):
        if fnmatch.fnmatch(file, pattern):
            file_path = topic_path + '/' + file
            topic = topic_path.split('/')[-1]
            tree = ET.parse(file_path)
            root = tree.getroot()
            dic_sentences, voc = get_sentences_of_file(root)
            events, entities = get_file_mention(root, file, dic_sentences, topic)
            event_mentions.extend(events)
            entity_mentions.extend(entities)
            vocab.update(voc)
            files.extend(get_tokens_from_file(root, file))


    return event_mentions, entity_mentions, files, vocab


def get_sentences_of_file(root):
    dict = {}
    vocab = set()

    sentence = []
    i = 0
    for child in root:
        sentence_num = child.attrib.get('sentence')
        if sentence_num == str(i):
            sentence.append([child.attrib.get('t_id'), child.text])
            vocab.add(child.text)
        else:
            if len(sentence) > 0:
                dict[str(i)] = sentence
            sentence = []
            if child.attrib.get('t_id') is not None:
                sentence.append([child.attrib.get('t_id'), child.text])
                vocab.add(child.text)
                i += 1


    return dict, vocab



def get_tokens_from_file(root, file_name):
    tokens = []
    sentence = 0
    for token in root:
        if token.tag == 'token':
            if int(token.attrib['sentence']) > sentence:
                tokens.append([])
                sentence += 1
            if sentence == 6:
                break

            sentence_lst = get_list_of_sentences(root)[sentence]
            token_id = get_token_id(sentence_lst, token.attrib['t_id'])
            tokens.append([file_name, token.attrib['sentence'], str(token_id), token.text, 'aa'])

    tokens.append([])
    return tokens


def get_index_of_word_in_sentence(word, sentence):
    return sentence.index(word)


def get_list_of_sentences(root):
    sentences = []
    sentence = []
    sent_id = 0
    for token in root:
        if token.tag == 'token':
            if int(token.attrib['sentence']) == sent_id:
                sentence.append([token.attrib['t_id'], token.text])
            elif int(token.attrib['sentence']) > sent_id:
                sentences.append(sentence)
                sent_id += 1
                sentence = [[token.attrib['t_id'], token.text]]

    sentences.append(sentence)

    return sentences


def get_token_id(sentence, t_id):
    for id, token_str in sentence:
        if t_id == id:
            return sentence.index([id, token_str])
    print('Error')
    return None




def get_tokens_ids(sentence, t_ids):
    sent_t_ids = [x[0] for x in sentence]
    return list(map(lambda x: sent_t_ids.index(x), t_ids))


def get_file_mention(root, file_name, sentences_text, topic):
    event_mentions = []
    entity_mentions = []
    mentions_dic = {}
    relation_mention_dic = {}

    sentences = get_list_of_sentences(root)

    for mention in root.find('Markables'):
        if mention.tag == 'ENTITY_MENTION' or mention.tag == 'EVENT_MENTION':
            mention_type = 'event' if mention.tag == 'EVENT_MENTION' else 'entity'
            m_id = mention.attrib['m_id']
            t_ids = []
            for term in mention:
                t_ids.append(term.attrib['t_id'])

            if len(t_ids) == 0:
                continue

            terms_ids = list(map(lambda x: int(x) - 1, t_ids))
            sentence = root[int(terms_ids[0])].attrib['sentence']
            tokens_ids = get_tokens_ids(sentences[int(sentence)], t_ids)


            if not args.entire_doc and int(sentence) >= 6: #only the fist 5 sentences are considered
                continue

            term = ' '.join(list(map(lambda x: root[x].text, terms_ids)))

            sentence_desc = ' '.join(x[1] for x in sentences_text[sentence])
            left = ' '.join(word for token_id, word in sentences_text[sentence] if int(token_id) < int(t_ids[0]))
            right = ' '.join(word for token_id, word in sentences_text[sentence] if int(token_id) > int(t_ids[-1]))

            is_pronoun = False
            tags = []
            if args.with_pos:
                doc = nlp(term)

                for token in doc:
                    tags.append(token.tag_)

                if len(tags) == 1 and (tags[0] == 'PRP' or tags[0] == 'PRPS'):
                    is_pronoun = True


            mentions_dic[m_id] = {
                    'doc_id': file_name,
                     'topic': topic,

                     'sent_id': int(sentence),
                     'm_id': m_id,
                     'tokens_number': tokens_ids, #terms_ids,
                     'event_entity': mention_type,
                     'tokens_str': term,
                     'tag': tags,

                     'full_sentence': sentence_desc,
                     'left_sentence': left,
                     'right_sentence': right,

                     'is_pronoun': is_pronoun,
                    'is_singleton': False,
                    'is_continuous': True,
                    'score': -1.0
                     }

        elif mention.tag  == 'ENTITY' or mention.tag == 'EVENT':
            m_id = mention.attrib['m_id']
            relation_mention_dic[m_id] = {
                'cluster_id': mention.attrib.get('instance_id', ''),
                'cluster_desc': mention.attrib.get('TAG_DESCRIPTOR', ''),
                'mention_type': mention.attrib.get('ent_type', '')
            }




    relation_source_target = {}
    relation_rid = {}

    for relation in root.find('Relations'):
        if relation.tag == 'REFERS_TO':
            target_mention = relation[-1].attrib['m_id']
            relation_rid[target_mention] = relation.attrib['r_id']
            for mention in relation:
                if mention.tag == 'source':
                    relation_source_target[mention.attrib['m_id']] = target_mention



    for mention, dic in mentions_dic.items():
        target = relation_source_target.get(mention, None)
        desc_cluster = ''
        type = ''


        if target is None or relation_mention_dic[target]['cluster_id'] == "":
            id_cluster = 'Singleton_' + dic['m_id'] + '_' +  dic['doc_id']
            #id_cluster = ""
        else:
            id_cluster = relation_mention_dic[target]['cluster_id']

        if target is not None:
            desc_cluster = relation_mention_dic[target]['cluster_desc']
            type = relation_mention_dic[target]['mention_type']


        mention_obj = dic.copy()
        mention_obj['cluster_id'] = id_cluster
        mention_obj['cluster_desc'] = desc_cluster


        #To adapt json format to Shany's system
        if type in ['PER', 'ORG', 'MIX']:
            mention_obj['mention_type'] = 'HUM'
        elif type == 'LOC':
            mention_obj['mention_type'] = 'LOC'
        elif mention_obj['event_entity'] == 'event':
            mention_obj['mention_type'] = 'ACT'
        else:
            mention_obj['mention_type'] = 'NON'

        #mention_obj['mention_type'] = type


        if mention_obj['event_entity'] == 'event':
            event_mentions.append(mention_obj)
        else:
            entity_mentions.append(mention_obj)


    return event_mentions, entity_mentions




def get_all_chains(mentions):
    chains = {}
    for mention_dic in mentions:
        chain_id = mention_dic['cluster_id']
        chains[chain_id] = [] if chain_id not in chains else chains[chain_id]
        chains[chain_id].append(mention_dic)

    return chains




def get_statistics(data_events, data_entities, data_desc, stat_file):
    docs = set()
    sentences = set()

    event_mentions_with_multiple_tokens = 0
    entity_mentions_with_multiple_tokens = 0


    for mention_dic in data_events:
        docs.add(mention_dic["doc_id"])
        sentences.add(mention_dic["doc_id"] + '_' + str(mention_dic["sent_id"]))
        if len(mention_dic['tokens_number']) > 1:
            event_mentions_with_multiple_tokens += 1

    for mention_dic in data_entities:
        docs.add(mention_dic["doc_id"])
        sentences.add(mention_dic["doc_id"] + '_' + str(mention_dic["sent_id"]))
        if len(mention_dic['tokens_number']) > 1:
            entity_mentions_with_multiple_tokens += 1




    event_chains = get_all_chains(data_events)
    entity_chains = get_all_chains(data_entities)
    event_singleton = len({id_cluster:mention for id_cluster, mention in event_chains.items() if len(mention) == 1})
    entity_singleton = len({id_cluster:mention for id_cluster, mention in entity_chains.items() if len(mention) == 1 })


    stat_file.write('\n')
    stat_file.write('Statistics on the {} set\n'.format(data_desc))
    stat_file.write('Docs: {}\n'.format(len(docs)))
    stat_file.write('Sentences: {}\n'.format(len(sentences)))
    stat_file.write('Event mentions: {}\n'.format(len(data_events)))
    stat_file.write('Entity mentions: {}\n'.format(len(data_entities)))
    stat_file.write('Event mentions with more than one token: {}\n'.format(event_mentions_with_multiple_tokens))
    stat_file.write('Entity mentions with more than one token: {}\n'.format(entity_mentions_with_multiple_tokens))
    stat_file.write('Event chains: {}\n'.format(len(event_chains)))
    stat_file.write('Event Singleton: {}\n'.format(event_singleton))
    stat_file.write('Entity chains: {}\n'.format(len(entity_chains)))
    stat_file.write('Entity Singleton: {}\n'.format(entity_singleton))
    stat_file.write('--------------------------------------\n')





if __name__ == '__main__':
    start = time.time()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print('Getting all mentions')
    train, dev, test, all, vocab = get_all_mention(args.data_path, args.output_dir)

    print('Getting mention statistics')
    stat_file = open(args.output_dir + '/statistics', 'w')
    get_statistics(train[0], train[1], 'train', stat_file)
    get_statistics(dev[0], dev[1], 'dev', stat_file)
    get_statistics(test[0], test[1], 'test', stat_file)
    get_statistics(all[0], all[1], 'all',  stat_file)
    stat_file.close()

    end = time.time() - start
    print('Time: {}'.format(end))