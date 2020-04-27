import argparse
from models import SpanScorer, SimplePairWiseClassifier, SpanEmbedder
import pyhocon
import json
from transformers import RobertaTokenizer, RobertaModel
from sklearn.cluster import AgglomerativeClustering
from itertools import product

from utils import *
from model_utils import *
from conll import write_output_file

logger = logging.getLogger(__name__)

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



def separate_doc_into_predicted_subtopics(ecb_texts, predicted_subtopics):
    '''
    Function to init the predicted subtopics as Shany Barhom
    :param ecb_texts:
    :param predicted_subtopics: Shany's file
    :return:
    '''
    text_by_subtopics = {}
    for i, doc_list in enumerate(predicted_subtopics):
        print(doc_list)
        if i not in text_by_subtopics:
            text_by_subtopics[i] = {}
        for doc in doc_list:
            if doc + '.xml' in ecb_texts:
                text_by_subtopics[i][doc + '.xml'] = ecb_texts[doc + '.xml']
    return text_by_subtopics




if __name__ == '__main__':
    config = pyhocon.ConfigFactory.parse_file(args.config_model_file)
    print(pyhocon.HOCONConverter.convert(config, "hocon"))
    device = 'cuda:{}'.format(config['gpu_num']) if torch.cuda.is_available() else 'cpu'

    # Load models and init clustering
    bert_model = RobertaModel.from_pretrained(config['roberta_model']).to(device)
    bert_model_hidden_size = bert_model.config.hidden_size

    span_repr = SpanEmbedder(config, bert_model_hidden_size, device).to(device)
    span_repr.load_state_dict(torch.load(config['span_repr_path'], map_location=device))
    span_repr.eval()
    span_scorer = SpanScorer(config, bert_model_hidden_size).to(device)
    span_scorer.load_state_dict(torch.load(config['span_scorer_path'], map_location=device))
    span_scorer.eval()
    pairwise_model = SimplePairWiseClassifier(config, bert_model_hidden_size).to(device)
    pairwise_model.load_state_dict(torch.load(config['pairwise_scorer'], map_location=device))
    pairwise_model.eval()

    clustering = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage=config['linkage_type'],
                                         distance_threshold=config['threshold'])

    # Load data
    with open(os.path.join(args.data_folder, '{}.json'.format(config['split'])), 'r') as f:
        ecb_texts = json.load(f)


    # relevant if using gold mentions
    with open(os.path.join(args.data_folder, '{}_entities.json'.format(config['split'])), 'r') as f:
        entity_mentions = json.load(f)
    entity_labels_dict = get_dict_labels(entity_mentions)

    with open(os.path.join(args.data_folder, '{}_events.json'.format(config['split'])), 'r') as f:
        event_mentions = json.load(f)
    event_labels_dict = get_dict_labels(event_mentions)

    '''
    Predicted subtopics from Shany
    '''

    # with open('/home/nlp/ariecattan/event_entity_coref_ecb_plus/data/external/document_clustering/predicted_topics', 'rb') as f:
    #     predicted_subtopics = pickle.load(f)
    #
    # ecb_texts_by_topic = separate_doc_into_predicted_subtopics(ecb_texts, predicted_subtopics)
    #


    ecb_texts_by_topic = separate_docs_into_topics(ecb_texts, subtopic=True)
    roberta_tokenizer = RobertaTokenizer.from_pretrained(config['roberta_model'])
    tokens = tokenize_set(ecb_texts_by_topic, roberta_tokenizer)
    all_topics, topic_list_of_docs, topic_origin_tokens, topic_bert_tokens, topic_start_end_bert = tokens



    doc_ids, sentence_ids, starts, ends = [], [], [], []
    all_topic_predicted_clusters = []
    max_cluster_id = 0

    # Go through each topic
    for t, topic in enumerate(all_topics):
        print('Processing topic {}'.format(topic))
        docs_embeddings, docs_length = pad_and_read_bert(topic_bert_tokens[t], bert_model)
        span_meta_data, span_embeddings, num_of_tokens = get_all_candidate_from_topic(
            config, topic_list_of_docs[t], topic_origin_tokens[t], topic_start_end_bert[t],
            docs_embeddings, docs_length)

        doc_id, sentence_id, start, end = span_meta_data
        start_end_embeddings, continuous_embeddings, width = span_embeddings

        if config['use_gold_mentions']:
            event_labels, entity_labels = get_candidate_labels(doc_id, start, end, event_labels_dict,
                                                                       entity_labels_dict)
            span_indices = event_labels.nonzero().squeeze(1)
            # event_labels = event_labels[event_span_indices]

        else:
            k = int(config['top_k'] * num_of_tokens)
            with torch.no_grad():
                span_emb = span_repr(start_end_embeddings, continuous_embeddings, width)
                span_scores = span_scorer(span_emb)
            _, span_indices = torch.topk(span_scores.squeeze(1), k, sorted=False)
            span_indices, _ = torch.sort(span_indices)

        number_of_mentions = len(span_indices)
        start_end_embeddings = start_end_embeddings[span_indices]
        continuous_embeddings = [continuous_embeddings[i] for i in span_indices]
        width = width[span_indices]
        torch.cuda.empty_cache()

        # Prepare all the pairs for the distance matrix
        first, second = zip(*list(product(range(len(span_indices)), repeat=2)))
        first = torch.tensor(first)
        second = torch.tensor(second)


        with torch.no_grad():
            g1 = span_repr(start_end_embeddings[first],
                           [continuous_embeddings[i] for i in first],
                           width[first])
            g2 = span_repr(start_end_embeddings[second],
                           [continuous_embeddings[i] for i in second],
                           width[second])
            pairwise_scores = torch.sigmoid(pairwise_model (g1, g2))


        # Affinity score to distance score
        pairwise_distances = 1 - pairwise_scores.view(number_of_mentions, number_of_mentions).detach().cpu().numpy()
        # identity_matrix = (~torch.eye(number_of_mentions).to(torch.bool)).to(torch.int)
        # pairwise_distances *= identity_matrix.numpy()

        predicted = clustering.fit(pairwise_distances)
        predicted_clusters = predicted.labels_ + max_cluster_id
        max_cluster_id = max(predicted_clusters) + 1

        doc_ids.extend(doc_id[span_indices.cpu()])
        sentence_ids.extend(sentence_id[span_indices].tolist())
        starts.extend(start[span_indices].tolist())
        ends.extend(end[span_indices].tolist())
        all_topic_predicted_clusters.extend(predicted_clusters)


    all_clusters = {}
    for i, cluster_id in enumerate(all_topic_predicted_clusters):
        if cluster_id not in all_clusters:
            all_clusters[cluster_id] = []
        all_clusters[cluster_id].append(i)

    print('Saving conll file...')
    doc_path = os.path.join(config['save_path'], '{}_events_{}_{}.predicted_conll'.format(
        config['split'], config['linkage_type'], config['threshold']))
    write_output_file(ecb_texts, all_clusters, doc_ids, starts, ends, doc_path)