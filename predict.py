import argparse
from models import MentionExtractor, SimplePairWiseClassifier, SimpleMentionExtractor
from utils import *
import pyhocon
import json
from transformers import RobertaTokenizer, RobertaModel
from sklearn.cluster import AgglomerativeClustering
from itertools import product
import math


parser = argparse.ArgumentParser()
parser.add_argument('--data_folder', type=str, default='data/ecb/mentions')
parser.add_argument('--config_model_file', type=str, default='configs/config_clustering.json')
args = parser.parse_args()

use_gold_label = False


def get_conll_predictions(data, doc_ids, starts, ends, cluster_dic):
    dic = {doc_id:{} for doc_id in set(doc_ids)}
    for i, (cluster_id, mentions) in enumerate(cluster_dic.items()):
        if len(mentions) == 1:
            continue

        for m in mentions:
            doc_id = doc_ids[m]
            start = starts[m]
            end = ends[m]

            single_token = start == end

            if single_token:
                dic[doc_id][start] = '(' + str(cluster_id) + ')'
            else:
                for i, token_id in enumerate(range(start, end + 1)):
                    if i == 0:
                        dic[doc_id][token_id] = '(' + str(cluster_id)
                    elif i == len(list(range(start, end + 1))) - 1:
                        dic[doc_id][token_id] = str(cluster_id) + ')'


    predicted_conll = []
    for doc_id, tokens in data.items():
        for sentence_id, token_id, token_text, flag, _ in tokens:
            if flag:
                dic_doc = dic[doc_id]
                cluster = dic_doc.get(token_id, '-')
                predicted_conll.append([doc_id, sentence_id, token_id, token_text, cluster])


    return predicted_conll




if __name__ == '__main__':
    config = pyhocon.ConfigFactory.parse_file(args.config_model_file)
    print(pyhocon.HOCONConverter.convert(config, "hocon"))
    device = 'cuda:{}'.format(config['gpu_num']) if torch.cuda.is_available() else 'cpu'

    # Load models
    bert_model_hidden_size = 768 if 'base' in config['roberta_model'] else 1024
    bert_model = RobertaModel.from_pretrained(config['roberta_model']).to(device)
    bert_model.eval()
    event_scorer = SimpleMentionExtractor(config, bert_model_hidden_size, device).to(device)
    event_scorer.load_state_dict(torch.load(config['event_mention_scorer'], map_location=device))
    event_scorer.eval()
    pairwise_scorer = SimplePairWiseClassifier(config, bert_model_hidden_size).to(device)
    pairwise_scorer.load_state_dict(torch.load(config['pairwise_scorer'], map_location=device))
    pairwise_scorer.eval()


    clustering = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage=config['linkage_type'],
                                         distance_threshold=config['threshold'])

    # Load data
    with open(os.path.join(args.data_folder, '{}_entities.json'.format(config['split'])), 'r') as f:
        entity_mentions = json.load(f)
    entity_labels_dict = get_dict_labels(entity_mentions)

    with open(os.path.join(args.data_folder, '{}_events.json'.format(config['split'])), 'r') as f:
        event_mentions = json.load(f)
    event_labels_dict = get_dict_labels(event_mentions)

    with open(os.path.join(args.data_folder, '{}.json'.format(config['split'])), 'r') as f:
        ecb_texts = json.load(f)

    ecb_texts_by_topic = separate_docs_into_topics(ecb_texts, subtopic=False)
    roberta_tokenizer = RobertaTokenizer.from_pretrained(config['roberta_model'], add_special_tokens=True)
    tokens = tokenize_set(ecb_texts_by_topic, roberta_tokenizer)
    all_topics, topic_list_of_docs, topic_origin_tokens, topic_bert_tokens, topic_start_end_bert = tokens



    doc_ids, sentence_ids, starts, ends = [], [], [], []
    all_topic_predicted_clusters = []
    max_cluster_id = 0

    # Go through each topic
    for t, topic in enumerate(all_topics):
        print('Processing topic {}'.format(topic))
        list_of_docs = topic_list_of_docs[t]
        docs_original_tokens = topic_origin_tokens[t]
        bert_tokens = topic_bert_tokens[t]
        docs_bert_start_end = topic_start_end_bert[t]

        docs_embeddings, docs_length = pad_and_read_bert(bert_tokens, bert_model, device)
        span_meta_data, span_embeddings, num_of_tokens = get_all_candidate_from_topic(
            config, device, list_of_docs, docs_original_tokens, docs_bert_start_end,
            docs_embeddings, docs_length, is_training=True)

        doc_id, sentence_id, start, end = span_meta_data
        topic_start_end_embeddings, topic_continuous_embeddings, topic_width = span_embeddings
        event_labels, entity_labels = get_candidate_labels(device, doc_id, start, end, event_labels_dict,
                                                                       entity_labels_dict)


        with torch.no_grad():
            event_span_embeddings, event_span_scores = event_scorer(topic_start_end_embeddings,
                                                                    topic_continuous_embeddings,
                                                                    topic_width)

        if config['use_gold_mentions']:
            event_span_indices = event_labels.nonzero().squeeze(1)
        else:
            event_span_scores, event_span_indices = torch.topk(event_span_scores, int(0.3 * num_of_tokens), sorted=False)
            event_span_indices, _ = torch.sort(event_span_indices)

        event_span_embeddings = event_span_embeddings[event_span_indices]
        event_labels = event_labels[event_span_indices]
        torch.cuda.empty_cache()


        if use_gold_label:
            predicted_clusters = event_labels

        else:
            first, second = zip(*list(product(range(len(event_span_indices)), repeat=2)))
            first = torch.tensor(first)
            second = torch.tensor(second)

            number_of_mentions = len(event_span_indices)


            with torch.no_grad():
                pairwise_scores = pairwise_scorer(event_span_embeddings[first], event_span_embeddings[second])
                pairwise_scores = torch.sigmoid(pairwise_scores)




            # Affinity score to distance score
            identity_matrix = (~torch.eye(number_of_mentions).to(torch.bool)).to(torch.int)
            pairwise_distances = 1 - pairwise_scores.view(number_of_mentions, number_of_mentions).detach().cpu().numpy()
            # pairwise_distances *= identity_matrix.numpy()


            predicted = clustering.fit(pairwise_distances)
            predicted_clusters = predicted.labels_ + max_cluster_id
            max_cluster_id = max(predicted_clusters)


        doc_ids.extend(doc_id[event_span_indices.cpu()])
        sentence_ids.extend(sentence_id[event_span_indices].tolist())
        starts.extend(start[event_span_indices].tolist())
        ends.extend(end[event_span_indices].tolist())
        all_topic_predicted_clusters.extend(predicted_clusters)




    all_clusters = {}
    for i, cluster_id in enumerate(all_topic_predicted_clusters):
        if cluster_id not in all_clusters:
            all_clusters[cluster_id] = []
        all_clusters[cluster_id].append(i)

    print('Saving conll file...')
    conll = get_conll_predictions(ecb_texts, doc_ids, starts, ends, all_clusters)

    doc_path = os.path.join(config['save_path'], '{}_events_{}_{}.predicted_conll'.format(
            config['split'], config['linkage_type'], config['threshold']))

    with open(doc_path, 'w') as f:
        f.write('#begin document {}_events'.format(config['split']) + '\n')
        for token in conll:
            f.write('\t'.join([str(x) for x in token]) + '\n')
        f.write('#end document')