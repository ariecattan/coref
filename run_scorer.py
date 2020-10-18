import sys
from coval.coval.conll import reader
from coval.coval.conll import util
from coval.coval.eval import evaluator
import pandas as pd
import os
from utils import *


def main():
    allmetrics = [('mentions', evaluator.mentions), ('muc', evaluator.muc),
            ('bcub', evaluator.b_cubed), ('ceafe', evaluator.ceafe),
            ('lea', evaluator.lea)]

    NP_only = 'NP_only' in sys.argv
    remove_nested = 'remove_nested' in sys.argv
    keep_singletons = ('remove_singletons' not in sys.argv
                       and 'removIe_singleton' not in sys.argv)
    min_span = False

    path = sys.argv[1]
    mention_type = sys.argv[2]
    sys_file = 'data/ecb/gold/dev_{}_topic_level.conll'.format(mention_type)

    all_scores = {}
    max_conll_f1 = (None, 0)

    for key_file in os.listdir(path):
        if key_file.endswith('conll'):
            print('Processing file: {}'.format(key_file))
            full_path = os.path.join(path,key_file)
            scores = evaluate(full_path, sys_file, allmetrics, NP_only, remove_nested,
                    keep_singletons, min_span)
            all_scores[key_file] = scores
            if scores['conll'] > max_conll_f1[1]:
                max_conll_f1 = (key_file, scores['conll'])

    df = pd.DataFrame.from_dict(all_scores)
    df.to_csv(os.path.join(path, 'all_scores.csv'))


    print(max_conll_f1)



def evaluate(key_file, sys_file, metrics, NP_only, remove_nested,
        keep_singletons, min_span):
    doc_coref_infos = reader.get_coref_infos(key_file, sys_file, NP_only,
            remove_nested, keep_singletons, min_span)

    conll = 0
    conll_subparts_num = 0

    scores = {}

    for name, metric in metrics:
        recall, precision, f1 = evaluator.evaluate_documents(doc_coref_infos,
                metric,
                beta=1)

        scores['{}_{}'.format(name, 'recall')] = recall
        scores['{}_{}'.format(name, 'precision')] = precision
        scores['{}_{}'.format(name, 'f1')] = f1

        if name in ["muc", "bcub", "ceafe"]:
            conll += f1
            conll_subparts_num += 1

        # print(name.ljust(10), 'Recall: %.2f' % (recall * 100),
        #         ' Precision: %.2f' % (precision * 100),
        #         ' F1: %.2f' % (f1 * 100))

    scores['conll'] = (conll / 3) * 100


    return scores

    # if conll_subparts_num == 3:
    #     conll = (conll / 3) * 100
    #     print('CoNLL score: %.2f' % conll)


if __name__ == '__main__':

    main()
