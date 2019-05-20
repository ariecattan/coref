from heapq import heappush, heappop

import spacy

from src.systems.data.data_utils import read_mentions_json_to_mentions_data_list


def clean_singletons(mentions):
    clusters = dict()
    for mention in mentions:
        if mention.coref_chain not in clusters:
            clusters[mention.coref_chain] = 0

        clusters[mention.coref_chain] += 1

    new_mentions = list()
    for mention in mentions:
        if clusters[mention.coref_chain] > 1:
            new_mentions.append(mention)

    return new_mentions


def main(entity_file, event_file):
    print('Done Loading all pairs, loading topics')
    all_mentions = read_mentions_json_to_mentions_data_list(event_file)
    all_mentions.extend(read_mentions_json_to_mentions_data_list(entity_file))

    print('Done Loading topics, create stats')
    new_mentions = clean_singletons(all_mentions)
    visualize_clusters(new_mentions)


def visualize_clusters(all_mentions):
    dispacy_obj = list()
    topic_contexts = dict()
    topic_cluster_ids = dict()

    for mention in all_mentions:
        if mention.topic_id not in topic_contexts:
            topic_contexts[mention.topic_id] = dict()
        context_mentions = topic_contexts[mention.topic_id]

        if mention.topic_id not in topic_cluster_ids:
            topic_cluster_ids[mention.topic_id] = {'event': 0, 'entity': 1}
        clustesr_ids = topic_cluster_ids[mention.topic_id]

        if mention.coref_chain not in clustesr_ids:
            if mention.mention_type == 'ACT':
                clustesr_ids[mention.coref_chain] = clustesr_ids['event']
                clustesr_ids['event'] += 1
            else:
                clustesr_ids[mention.coref_chain] = clustesr_ids['entity']
                clustesr_ids['entity'] += 1

        mention_int_cluster_id = clustesr_ids[mention.coref_chain]

        context, start, end = get_context_start_end(mention)

        if context not in context_mentions:
            context_mentions[context] = list()

        heappush(context_mentions[context], (start, end, mention.mention_type, str(mention_int_cluster_id)))

    print(str(topic_cluster_ids))
    for topic_id, context_mentions in topic_contexts.items():
        cluster_context = ""
        ents = list()
        for context, mentions_heap in context_mentions.items():
            for i in range(len(mentions_heap)):
                ment_pair = heappop(mentions_heap)
                real_start = len(cluster_context) + 1 + ment_pair[0]
                real_end = len(cluster_context) + 1 + ment_pair[1]
                ent_label = ment_pair[3]
                if ment_pair[2] == 'ACT':
                    ents.append({'start': real_start, 'end': real_end, 'label': ent_label + '_EVENT'})
                else:
                    ents.append({'start': real_start, 'end': real_end, 'label': ent_label + '_ENTITY'})

            cluster_context = cluster_context + '\n' + context

        dispacy_obj.append({
            'text': cluster_context,
            'ents': ents,
            'title': topic_id
        })

    colors = {
        '1_ENTITY': 'red', '1_EVENT': 'steelblue',
        '2_ENTITY': 'firebrick', '2_EVENT': 'violet',
        '3_ENTITY': 'wheat', '3_EVENT': 'gold',
        '4_ENTITY': 'coral', '4_EVENT': 'plum',
        '5_ENTITY': 'peru', '5_EVENT': 'deepskyblue',
        '6_ENTITY': 'silver', '6_EVENT': 'deeppink',
        '7_ENTITY': 'lime', '7_EVENT': 'hotpink',
        '8_ENTITY': 'c', '8_EVENT': 'brown',
        '9_ENTITY': 'gray', '9_EVENT': 'lightpink',
        '10_ENTITY': 'tomato', '10_EVENT': 'tan',
        '11_ENTITY': 'khaki', '11_EVENT': 'olive',
        '12_ENTITY': 'goldenrod', '12_EVENT': 'teal',
        '13_ENTITY': 'linen', '13_EVENT': 'seagreen',
        '14_ENTITY': 'cornsilk', '14_EVENT': 'palegreen',
        '15_ENTITY': 'orange', '15_EVENT': 'olivedrab'
    }
    options = {'ents': ['1_ENTITY', '1_EVENT', '2_ENTITY', '2_EVENT',
                        '3_ENTITY', '3_EVENT', '4_ENTITY', '4_EVENT',
                        '5_ENTITY', '5_EVENT', '6_EVENT', '6_ENTITY',
                        '7_EVENT', '7_ENTITY', '8_EVENT', '8_ENTITY',
                        '9_EVENT', '9_ENTITY', '10_EVENT', '10_ENTITY',
                        '11_EVENT', '11_ENTITY', '12_EVENT', '12_ENTITY',
                        '13_EVENT', '13_ENTITY', '14_EVENT', '14_ENTITY',
                        '15_EVENT', '15_ENTITY'], 'colors': colors}

    spacy.displacy.serve(dispacy_obj, style='ent', manual=True, options=options)


def get_context_start_end(mention):
    start = -1
    end = -1
    context = ""
    for i in range(len(mention.mention_context)):
        if i == mention.tokens_number[0]:
            start = len(context)

        if i == 0:
            context = mention.mention_context[i]
        else:
            context = context + ' ' + mention.mention_context[i]

        if i == int(mention.tokens_number[-1]):
            end = len(context)

    return context, start, end


if __name__ == '__main__':
    _event_file = 'data/interim/kian/gold_mentions_with_context/ECB_Dev_Event_gold_mentions.json'
    _entity_file = 'data/interim/kian/gold_mentions_with_context/ECB_Dev_Entity_gold_mentions.json'
    main(_entity_file, _event_file)
