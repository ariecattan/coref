from sklearn.cluster import AgglomerativeClustering
import argparse
import pyhocon
from transformers import AutoTokenizer, AutoModel
from itertools import product
import collections
from tqdm import tqdm

from conll import write_output_file
from models import SpanScorer, SimplePairWiseClassifier, SpanEmbedder
from utils import *
from model_utils import *



def init_models(config, device, model_num):
    span_repr = SpanEmbedder(config, device).to(device)
    span_repr.load_state_dict(torch.load(os.path.join(config['model_path'],
                                                      "span_repr_{}".format(model_num)),
                                         map_location=device))
    span_repr.eval()
    span_scorer = SpanScorer(config).to(device)
    span_scorer.load_state_dict(torch.load(os.path.join(config['model_path'],
                                                        "span_scorer_{}".format(model_num)),
                                           map_location=device))
    span_scorer.eval()
    pairwise_scorer = SimplePairWiseClassifier(config).to(device)
    pairwise_scorer.load_state_dict(torch.load(os.path.join(config['model_path'],
                                                           "pairwise_scorer_{}".format(model_num)),
                                              map_location=device))
    pairwise_scorer.eval()

    return span_repr, span_scorer, pairwise_scorer




def is_included(docs, starts, ends, i1, i2):
    doc1, start1, end1 = docs[i1], starts[i1], ends[i1]
    doc2, start2, end2 = docs[i2], starts[i2], ends[i2]

    if doc1 == doc2 and (start1 >= start2 and end1 <= end2):
        return True
    return False


def remove_nested_mentions(cluster_ids, doc_ids, starts, ends):
    # nested_mentions = collections.defaultdict(list)
    # for i, x in range(len(cluster_ids)):
    #     nested_mentions[x].append(i)

    doc_ids = np.asarray(doc_ids)
    starts = np.asarray(starts)
    ends = np.asarray(ends)

    new_cluster_ids, new_docs_ids, new_starts, new_ends = [], [], [], []

    for cluster, idx in cluster_ids.items():
        docs = doc_ids[idx]
        start = starts[idx]
        end = ends[idx]


        for i in range(len(idx)):
            indicator = [is_included(docs, start, end, i, j) for j in range(len(idx))]
            if sum(indicator) > 1:
                continue

            new_cluster_ids.append(cluster)
            new_docs_ids.append(docs[i])
            new_starts.append(start[i])
            new_ends.append(end[i])


    clusters = collections.defaultdict(list)
    for i, cluster_id in enumerate(new_cluster_ids):
        clusters[cluster_id].append(i)

    return clusters, new_docs_ids, new_starts, new_ends






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config_clustering.json')
    args = parser.parse_args()


    config = pyhocon.ConfigFactory.parse_file(args.config)
    print(pyhocon.HOCONConverter.convert(config, "hocon"))
    create_folder(config['save_path'])
    device = 'cuda:{}'.format(config['gpu_num'][0]) if torch.cuda.is_available() else 'cpu'


    # Load models and init clustering
    bert_model = AutoModel.from_pretrained(config['bert_model']).to(device)
    config['bert_hidden_size'] = bert_model.config.hidden_size



    bert_tokenizer = AutoTokenizer.from_pretrained(config['bert_model'])
    data = create_corpus(config, bert_tokenizer, 'dev')

    clustering_5 = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage=config['linkage_type'],
                                         distance_threshold=0.5)
    clustering_55 = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage=config['linkage_type'],
                                         distance_threshold=0.55)
    clustering_6 = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage=config['linkage_type'],
                                         distance_threshold=0.6)
    clustering_65 = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage=config['linkage_type'],
                                         distance_threshold=0.65)
    clustering_7 = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage=config['linkage_type'],
                                         distance_threshold=0.7)

    clustering = []

    for x in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
        agglo = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage=config['linkage_type'],
                                         distance_threshold=x)

        clustering.append(agglo)


    for num in range(10):
        print('Model {}'.format(num))
        span_repr, span_scorer, pairwise_scorer = init_models(config, device, num)

        clusters = [list(), list(), list(), list(), list(), list(), list()]
        max_ids = [0, 0, 0, 0, 0, 0, 0]
        threshold = {id: thresh for id, thresh in enumerate([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8])}

        doc_ids, sentence_ids, starts, ends = [], [], [], []

        for topic_num, topic in enumerate(data.topic_list):
            print('Processing topic {}'.format(topic))
            docs_embeddings, docs_length = pad_and_read_bert(data.topics_bert_tokens[topic_num], bert_model)
            span_meta_data, span_embeddings, num_of_tokens = get_all_candidate_from_topic(
                config, data, topic_num, docs_embeddings, docs_length)

            doc_id, sentence_id, start, end = span_meta_data
            start_end_embeddings, continuous_embeddings, width = span_embeddings
            width = width.to(device)

            labels = data.get_candidate_labels(doc_id, start, end)

            if config['use_gold_mentions']:
                span_indices = torch.nonzero(labels).squeeze(1)
            else:
                with torch.no_grad():
                    span_emb = span_repr(start_end_embeddings, continuous_embeddings, width)
                    span_scores = span_scorer(span_emb)

                if config.exact:
                    span_indices = torch.where(span_scores > 0.5)[0]
                else:
                    k = int(config['top_k'] * num_of_tokens)
                    _, span_indices = torch.topk(span_scores.squeeze(1), k, sorted=False)
                # span_indices, _ = torch.sort(span_indices)

            number_of_mentions = len(span_indices)
            start_end_embeddings = start_end_embeddings[span_indices]
            continuous_embeddings = [continuous_embeddings[i] for i in span_indices]
            width = width[span_indices]
            torch.cuda.empty_cache()

            # Prepare all the pairs for the distance matrix
            first, second = zip(*list(product(range(len(span_indices)), repeat=2)))
            first = torch.tensor(first)
            second = torch.tensor(second)

            torch.cuda.empty_cache()
            all_scores = []
            with torch.no_grad():
                for i in range(0, len(first), 10000):
                    # end_max = min(i+100000, len(first))
                    end_max = i + 10000
                    first_idx, second_idx = first[i:end_max], second[i:end_max]
                    g1 = span_repr(start_end_embeddings[first_idx],
                                   [continuous_embeddings[k] for k in first_idx],
                                   width[first_idx])
                    g2 = span_repr(start_end_embeddings[second_idx],
                                   [continuous_embeddings[k] for k in second_idx],
                                   width[second_idx])
                    scores = pairwise_scorer(g1, g2)

                    torch.cuda.empty_cache()
                    if config['training_method'] in ('continue', 'e2e') and not config['use_gold_mentions']:
                        g1_score = span_scorer(g1)
                        g2_score = span_scorer(g2)
                        scores += g1_score + g2_score

                    scores = torch.sigmoid(scores)
                    all_scores.extend(scores.detach().cpu().squeeze(1))
                    torch.cuda.empty_cache()

            all_scores = torch.stack(all_scores)

            # Affinity score to distance score
            pairwise_distances = 1 - all_scores.view(number_of_mentions, number_of_mentions).numpy()

            doc_ids.extend(doc_id[span_indices.cpu()])
            sentence_ids.extend(sentence_id[span_indices].tolist())
            starts.extend(start[span_indices].tolist())
            ends.extend(end[span_indices].tolist())


            for i, agglomerative in enumerate(clustering):
                predicted = agglomerative.fit(pairwise_distances)
                predicted_clusters = predicted.labels_ + max_ids[i]
                max_ids[i] = max(predicted_clusters) + 1
                clusters[i].extend(predicted_clusters)


        for i, predicted in enumerate(clusters):
            print('Saving cluster for threshold {}'.format(threshold[i]))
            all_clusters = collections.defaultdict(list)
            for span_id, cluster_id in enumerate(predicted):
                all_clusters[cluster_id].append(span_id)

            if not config['use_gold_mentions']:
                all_clusters, new_doc_ids, new_starts, new_ends = remove_nested_mentions(all_clusters, doc_ids, starts, ends)
            else:
                new_doc_ids, new_starts, new_ends = doc_ids, starts, ends

            # removing singletons
            all_clusters = {cluster_id:mentions for cluster_id, mentions in all_clusters.items()
                               if len(mentions) > 1}

            # print('Saving conll file...')
            doc_name = 'dev_{}_model_{}_{}_{}'.format(
                config['mention_type'], num, config['linkage_type'], threshold[i])

            write_output_file(data.documents, all_clusters, new_doc_ids, new_starts, new_ends, config['save_path'], doc_name,
                              topic_level=config.topic_level, corpus_level=not config.topic_level)

