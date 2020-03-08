import json
import os, fnmatch
import argparse
import allennlp

parser = argparse.ArgumentParser(description='Parse conll data')
parser.add_argument('--path_to_conll', type=str, default='data/datasets/conll-2012/v4/data')
args = parser.parse_args()


def check_file_contains_coref(doc):
    for token in doc:
        if token[-1] != '-':
            return True
    return False

def read_file(conll_file_path):
    data = []
    with open(conll_file_path, 'r') as f:
        for line in f:
            if line.startswith('#begin'):
                doc = []
            elif line.startswith('#end'):
                data.append(doc)

            elif len(line) > 0 and line != '\n':
                doc.append(line.split())
            else:
                continue
    return data


def read_folder(dataset):
    pattern = '*gold_conll'
    all_docs = []
    folder_annotation = os.path.join(dataset, 'data', 'english', 'annotations')
    for genre in os.listdir(folder_annotation):
        print('Processing genre: {}'.format(genre))
        for sub_genre in os.listdir(os.path.join(folder_annotation, genre)):
            for num in os.listdir(os.path.join(folder_annotation, genre, sub_genre)):
                for file in os.listdir(os.path.join(folder_annotation, genre, sub_genre, num)):
                    if fnmatch.fnmatch(file, pattern):
                        file_data = read_file(os.path.join(folder_annotation, genre, sub_genre, num, file))
                        if file_data:
                            all_docs.extend(file_data)

    print('{} files were preprocessed'.format(len(all_docs)))
    return all_docs


def clean_cluster(cluster):
    cl = cluster
    if cluster.startswith('('):
        cl = cluster[1:]
    if cluster.endswith(')'):
        cl = cl[:-1]
    return cl



def get_clusters(doc):
    clusters = {}
    i = 0

    while i < len(doc):
        token = doc[i]
        coref_chain = token[-1]
        i += 1

        if coref_chain.startswith('(') and coref_chain.endswith(')') and '|' not in coref_chain:
            cluster = coref_chain[1:-1]
            clusters[cluster] = [] if cluster not in clusters else clusters[cluster]
            clusters[cluster].append(token)


        elif coref_chain.startswith('(') and '|' not in coref_chain:
            mention = []
            cluster = coref_chain[1:]
            j = i-1

            while True:
                mention.append(doc[j])
                if cluster + ')' in doc[j][-1] or j == len(doc) - 1:
                    clusters[cluster] = [] if cluster not in clusters else clusters[cluster]
                    clusters[cluster].append(mention)
                    break
                j += 1


        elif coref_chain.startswith('(') and '|' in coref_chain:
            cls = [clean_cluster(x) for x in coref_chain.split('|')]

            for cluster in cls:
                mention = []
                j = i-1
                while j < len(doc):
                    mention.append(doc[j])
                    if cluster + ')' in doc[j][-1]:
                        clusters[cluster] = [] if cluster not in clusters else clusters[cluster]
                        clusters[cluster].append(mention)
                        break
                    j += 1



    return clusters


def build_clusters(docs):
    clusters = {}
    num_of_clusters = 0
    num_of_mentions = 0
    for doc in docs:
        doc_name = doc[0][0] + '_' + doc[0][1]
        cls = get_clusters(doc)
        num_of_clusters += len(cls)
        for cl, mentions in cls.items():
            num_of_mentions += len(mentions)
            clusters[doc_name + '_' + cl] = mentions

    return clusters


def compute_stat(clusters):
    num_of_verb = 0

    for cluster, mentions in clusters.items():
        event = False
        for mention in mentions:
            if len(mention) > 4 and 'VB' in mention[4]:
                event = True
                break
        num_of_verb += event

    return num_of_verb


def avg_words_in_sentence(docs):
    sentences = 0
    num_of_tokens = 0
    for doc in docs:
        num_of_tokens += len(doc)
        for token in doc:
            if token[2] == '0':
                sentences += 1

    return num_of_tokens / sentences


if __name__ == '__main__':
    train_folder = os.path.join(args.path_to_conll, 'train')
    dev_folder = os.path.join(args.path_to_conll, 'development')
    test_folder = os.path.join(args.path_to_conll, 'test')
    print('Processing training set')
    train_docs = read_folder(train_folder)
    print('Processing dev set')
    dev_docs = read_folder(dev_folder)
    print('Processing test set')
    test_docs = read_folder(test_folder)

    print('Building coref chains on the train set')
    train_clusters = build_clusters(train_docs)
    print('{} clusters in the train set'.format(len(train_clusters)))
    print('{} mentions in the train set'.format(sum(len(mentions) for cl, mentions in train_clusters.items())))
    print('Num of verbs:{}'.format(compute_stat(train_clusters)))

    print('Building coref chains on the dev set')
    dev_clusters = build_clusters(dev_docs)
    print('{} clusters in the dev set'.format(len(dev_clusters)))
    print('{} mentions in the dev set'.format(sum(len(mentions) for cl, mentions in dev_clusters.items())))
    print('Num of verbs:{}'.format(compute_stat(dev_clusters)))

    print('Building coref chains on the test set')
    test_clusters = build_clusters(test_docs)
    print('{} clusters in the test set'.format(len(test_clusters)))
    print('{} mentions in the test set'.format(sum(len(mentions) for cl, mentions in test_clusters.items())))
    print('Num of verbs:{}'.format(compute_stat(test_clusters)))

    all_docs = train_docs + dev_docs + test_docs
    print('Average tokens in sentence: {}'.format(avg_words_in_sentence(all_docs)))


