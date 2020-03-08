import os
import argparse
import json




def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_token_cluster_id_map(mentions):
    map_token_cluster_id = {}
    for mention in mentions:
        cluster_id = str(mention['cluster_id'])
        singleton_flag = mention['singleton']
        doc_id = mention['doc_id']
        continuous_tokens = list(range(min(mention['tokens_ids']), max(mention['tokens_ids']) + 1))

        for i, token_id in enumerate(continuous_tokens):
            bie_tag = 'be'
            if len(continuous_tokens) > 1:
                if i == 0:
                    bie_tag = 'b'
                elif i == len(continuous_tokens) - 1:
                    bie_tag = 'e'
                else:
                    bie_tag = 'i'

            token_key = doc_id + '_' + str(token_id)
            map_token_cluster_id[token_key] = [] if token_key not in map_token_cluster_id else map_token_cluster_id[token_key]
            map_token_cluster_id[token_key].append([cluster_id, bie_tag, singleton_flag])
    return map_token_cluster_id



def get_conll_gold_file(ecb_tokens, map_token_cluster_id, with_singleton):
    ecb_tokens_with_conll = []
    for token in ecb_tokens:
        doc_id, sentence_id, token_id, token_text, flag_sentence, flag_continuous = token
        if not flag_sentence:
            continue
        cluster_id, bie_tag, singleton_flag = '-', '-', '-'
        token_key = doc_id + '_' + str(token_id)
        if token_key in map_token_cluster_id:
            cluster_id = ''
            for i, (c_id, b_tag, s_tag) in enumerate(map_token_cluster_id[token_key]):
                flag_sep = cluster_id != ''
                if with_singleton or not s_tag:
                    if b_tag == 'be':
                        cluster_id += '|' if flag_sep else ''
                        cluster_id += '(' + c_id + ')'
                    elif b_tag == 'b':
                        cluster_id += '|' if flag_sep else ''
                        cluster_id += '(' + c_id
                    elif b_tag == 'e':
                        cluster_id += '|' if flag_sep else ''
                        cluster_id += c_id + ')'
            if cluster_id == '':
                cluster_id = '-'

        ecb_tokens_with_conll.append([doc_id, sentence_id, token_id, token_text, cluster_id])

    return ecb_tokens_with_conll




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--with_singleton', type=str2bool, default=True)
    parser.add_argument('--data_dir', type=str, default='data/ecb/mentions')
    parser.add_argument('--output_dir', type=str, default='data/ecb/gold')
    args = parser.parse_args()

    if args.with_singleton:
        args.output_dir = os.path.join(args.output_dir, 'with_singleton')
    else:
        args.output_dir = os.path.join(args.output_dir, 'without_singleton')

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    for sp in ['train', 'dev', 'test']:
        print('Processing {}'.format(sp))
        with open(os.path.join(args.data_dir, sp + '.json')) as f:
            data = json.load(f)
        raw_data = []
        for doc, tokens in data.items():
            raw_data.extend([[doc] + tok for tok in tokens])

        for i, mention_type in enumerate(['_events', '_entities']):
            with open(os.path.join(args.data_dir, sp + mention_type + '.json'), 'r') as f:
                mentions = json.load(f)
            map_token_cluster_id = get_token_cluster_id_map(mentions)
            conll_data = get_conll_gold_file(raw_data, map_token_cluster_id, args.with_singleton)

            output_path = os.path.join(args.output_dir, sp + mention_type + '.gold_conll')
            print('Creating file: {}'.format(output_path))
            with open(output_path, 'w') as f:
                f.write('#begin document ' + sp + mention_type + '\n')
                for token in conll_data:
                    f.write('\t'.join([str(x) for x in token]) + '\n')
                f.write('#end document')

