import argparse
import xml.etree.ElementTree as ET
import os, fnmatch


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='ECB+_LREC2014/ECB+', help='ECB+ data')
args = parser.parse_known_args()[0]


def read_file(file_path):
    """
    function to go over the file and extract only the sentences
    :param file_path:
    :return: doc_id, subtopic (either ECB or ECB+) and list of sentences
    """
    sentences = []
    tree = ET.parse(file_path)
    root = tree.getroot()
    subtopic = 'ECB' if file_path.endswith('ecb.xml') else 'ECB+'
    sentence = []
    i = 0
    for child in root:
        if child.attrib.get('sentence') == str(i):
            sentence.append([child.attrib.get('t_id'), child.text])
        else:
            if len(sentence) > 0:
                sentences.append(sentence)
            sentence = []
            if child.attrib.get('t_id') is not None:
                sentence.append([child.attrib.get('t_id'), child.text])
                i += 1

    return root.attrib['doc_id'], subtopic, sentences



def read_folder(folder_path):
    """
    read all xml files of a folder
    :param folder_path:
    :return: dictionary of the two subtopics
    """
    topic = {}
    topic['ECB'] = []
    topic['ECB+'] = []
    pattern = "*.xml"

    for file in os.listdir(folder_path):
        if fnmatch.fnmatch(file, pattern):
            doc_id, subtopic, sentences = read_file(folder_path + '/' + file)
            topic[subtopic].append([file, doc_id, sentences])

    return topic

def main():
    corpus_path = args.data
    corpus = []
    for topic_folder in os.scandir(corpus_path):
        if topic_folder.is_dir():
            topic = read_folder(corpus_path + '/' + topic_folder.name)
            corpus.append([int(topic_folder.name), topic])

    corpus.sort(key=lambda x: x[0])
    return corpus


if __name__ == '__main__':
    corpus = main()