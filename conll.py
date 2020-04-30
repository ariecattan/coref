import collections
import operator
import os



def get_dict_map(predictions, doc_ids, starts, ends):
  doc_start_map = collections.defaultdict(list)
  doc_end_map = collections.defaultdict(list)
  doc_word_map = collections.defaultdict(list)

  for cluster_id, mentions in predictions.items():
      for idx in mentions:
          doc_id, start, end = doc_ids[idx], starts[idx], ends[idx]
          start_key = doc_id + '_' + str(start)
          end_key = doc_id + '_' + str(end)
          if start == end:
              doc_word_map[start_key].append(cluster_id)
          else:
              doc_start_map[start_key].append((cluster_id, end))
              doc_end_map[end_key].append((cluster_id, start))

  for k, v in doc_start_map.items():
      doc_start_map[k] = [cluster_id for cluster_id, end_key in sorted(v, key=operator.itemgetter(1), reverse=True)]
  for k, v in doc_end_map.items():
      doc_end_map[k] = [cluster_id for cluster_id, end_key in sorted(v, key=operator.itemgetter(1), reverse=True)]

  return doc_start_map, doc_end_map, doc_word_map



def output_conll(data, doc_word_map, doc_start_map, doc_end_map):
    predicted_conll = []
    for doc_id, tokens in data.items():
        for sentence_id, token_id, token_text, flag, _ in tokens:
            clusters = '-'
            coref_list = list()
            if flag:
                token_key = doc_id + '_' + str(token_id)
                if token_key in doc_word_map:
                    for cluster_id in doc_word_map[token_key]:
                        coref_list.append('({})'.format(cluster_id))
                if token_key in doc_start_map:
                    for cluster_id in doc_start_map[token_key]:
                        coref_list.append('({}'.format(cluster_id))
                if token_key in doc_end_map:
                    for cluster_id in doc_end_map[token_key]:
                        coref_list.append('{})'.format(cluster_id))

            if len(coref_list) > 0:
                clusters = '|'.join(coref_list)

            predicted_conll.append([doc_id, sentence_id, token_id, token_text, clusters])


    return predicted_conll



def write_output_file(data, predictions, doc_ids, starts, ends, path):
    doc_start_map, doc_end_map, doc_word_map = get_dict_map(predictions, doc_ids, starts, ends)
    conll = output_conll(data, doc_word_map, doc_start_map, doc_end_map)

    doc_name = '_'.join(os.path.basename(path).split('_')[:2])
    with open(path, 'w') as f:
        f.write('#begin document {}\n'.format(doc_name))
        for token in conll:
            f.write('\t'.join([str(x) for x in token]) + '\n')
        f.write('#end document')