import argparse
import json
import matplotlib.pyplot as plt
import operator

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='meantime_data',
                    help=' The path data folder')

args = parser.parse_args()


def get_mentions_of_cluster(dataset, cluster_id):
    mentions = []
    for mention in dataset:
        if mention['coref_chain'] == cluster_id:
            mentions.append(mention)

    return mentions


def get_all_chains(mentions):
    clusters = {}
    for mention_dic in mentions:
        chain = mention_dic['coref_chain']
        clusters[chain] = [] if chain not in clusters else clusters[chain]
        clusters[chain].append(mention_dic)

    return clusters


def get_cluster_by_mention_num(clusters, num):
    clusters_names = []
    for cluster, doc_mention in clusters.items():
        num_of_mentions = len(doc_mention)
        if num_of_mentions == num:
            clusters_names.append(cluster)

    return clusters_names


def get_gold_within_doc(mentions):
    wd_cluster = {}
    for mention in mentions:
        chain = mention['coref_chain']
        doc = mention['doc_id']
        id_within_doc = chain + '_' + doc
        wd_cluster[id_within_doc] = [] if id_within_doc not in wd_cluster else wd_cluster[id_within_doc]
        wd_cluster[id_within_doc].append(mention)

    return wd_cluster



def get_metainfo(clusters):
    """
    print num of mentions per clusters
    :param clusters:
    :return:
    """
    dic = {}
    for cluster, doc_mention in clusters.items():
        num_of_mentions = len(doc_mention)
        dic[num_of_mentions] = dic.get(num_of_mentions, 0) + 1

    for length, num_of_clusters in sorted(dic.items()):
        print("There are {} clusters with {} mentions".format(num_of_clusters, length))

    number = dic.values()
    labels = dic.keys()

    #get_pie_chart(number, labels)

def extract_mention_text(cluster):
    mentions = []
    for mention in cluster:
        mention.append(mention['MENTION_TEXT'])
    return mentions


def get_pie_chart(values, labels):
    patches, texts = plt.pie(values, shadow=True, startangle=90)
    plt.legend(patches, labels, loc="best")
    plt.axis('equal')
    plt.show()


def within_to_cross(within_doc_cluster):
    cross_doc = {}
    for within in within_doc_cluster:
        name = within.split('_')[0]
        if name != 'INTRA' and name != 'Singleton':
            cross_doc[name] = [] if name not in cross_doc else cross_doc[name]
            cross_doc[name].append(within)

    return cross_doc


def find_most_popular_word(clusters, within_doc_cluster):
    words = {}
    for cluster in clusters:
        mentions = within_doc_cluster[cluster]
        vocab = set()
        for mention in mentions:
            text = mention['MENTION_TEXT']
            vocab.add(text)

        for word in vocab:
            words[word] = words.get(word, 0) + 1

    most_word = max(words.items(), key=operator.itemgetter(1))
    return most_word[0], most_word[1]/len(clusters)




def get_prob(within_doc_cluster):
    cross_doc = within_to_cross(within_doc_cluster)
    length = 0
    prob = 0
    for cluster, within in cross_doc.items():
        word, coverage = find_most_popular_word(within, within_doc_cluster)
        length += len(within)
        prob += coverage * len(within)

    return prob / length



if __name__ == '__main__':
    data = args.data

    with open(data + '/all_entity_gold_mentions.json', 'r') as f:
        entity_mentions = json.load(f)

    with open(data + '/all_event_gold_mentions.json', 'r') as f:
        event_mentions = json.load(f)


    event_clusters = get_all_chains(event_mentions)
    event_within_clusters = get_gold_within_doc(event_mentions)

    entity_cross_clusters = get_all_chains(entity_mentions)
    entity_within_clusters = get_gold_within_doc(entity_mentions)
