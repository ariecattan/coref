import argparse
import os
import pickle
from allennlp.predictors.predictor import Predictor
import logging
import json
from collections import defaultdict
import torch
from tqdm import tqdm


def get_root_of_mention(root, mention):
    for child in root['children']:
        if mention == child['word']:
            return child
        elif mention in child['word']:
            root = child

    print('No root matched for the mention: {}'.format(mention))
    return None



def main(args, predictor):
    if args.raw_data:
        logging.info('Sentence level constituency trees')
        num_sentences = 0
        for topic in os.listdir(args.raw_data):
            constituency_tree = {}
            logger.info('Processed topic {}'.format(topic))
            with open(os.path.join(args.raw_data, topic), 'r') as f_r:
                for line in f_r:
                    if not line.strip():
                        continue
                    elif line.strip().endswith('xml'):
                        corpus_file = line.strip()
                        constituency_tree[corpus_file] = []
                        logger.info('Processed file: {}'.format(corpus_file))
                    else:
                        logger.info('Processed sentence num: {}'.format(line.split(' ')[0]))
                        sent = ' '.join(line.split(' ')[1:])
                        tree = predictor.predict_json({"sentence": sent})
                        constituency_tree[corpus_file].append(tree['hierplane_tree']['root'])
                        num_sentences += 1

            with open(os.path.join(args.output_path, topic), 'wb') as f_w:
                pickle.dump(constituency_tree, f_w)

        logger.info('{} sentences were processed'.format(num_sentences))


    if args.mention_path:
        logging.info('Mention level constituency trees')
        with open(args.mention_path, 'r') as f:
            mentions = json.load(f)

        mention_str = [ {"sentence" :  mention['tokens_str']} for mention in mentions]
        keys = [mention['mention_id'] for mention in mentions]
        constituency_trees = []
        for i in tqdm(range(0, len(mention_str), args.batch_size)):
            constituency_trees.extend(predictor.predict_batch_json(mention_str[i:i+args.batch_size]))
        trees = {key:tree['hierplane_tree']['root'] for key, tree in zip(keys, constituency_trees)}
        logger.info('{} mentions were processed'.format(len(trees) + 1))

        with open(os.path.join(args.output_path), 'wb') as f:
            pickle.dump(trees, f)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--raw_data', type=str, default='')
    parser.add_argument('--mention_path', type=str, default='wiki_gold_mentions_json/new_files/WEC_Train_Event_gold_mentions.json')
    parser.add_argument('--output_path', type=str, default='wiki_gold_mentions_json/constituency_tree/new_files/WEC_Train_constituency_trees')
    parser.add_argument('--gpu', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=1024)
    args = parser.parse_args()


    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                        filename=os.path.join('wiki_gold_mentions_json', 'constituency_tree', "log.txt"),
                        level=logging.DEBUG, filemode='w', datefmt='%Y-%m-%d %H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler())
    logger = logging.getLogger(__name__)

    cuda_device = args.gpu if torch.cuda.is_available() else -1
    predictor = Predictor.from_path(
        "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo-constituency-parser-2018.03.14.tar.gz",
        cuda_device=cuda_device)
    logger.info('Using GPU {}'.format(cuda_device))

    main(args, predictor)



    '''
    python src/preprocessing/auto_mention_extraction.py --mention_path wiki_gold_mentions_json/clean_WEC/CleanWEC_Dev_Event_gold_mentions.json --output_path wiki_gold_mentions_json/constituency_tree/clean_WEC/CleanWEC_Dev_constituency_trees
    python src/preprocessing/auto_mention_extraction.py --mention_path wiki_gold_mentions_json/clean_WEC/CleanWEC_Test_Event_gold_mentions.json --output_path wiki_gold_mentions_json/constituency_tree/clean_WEC/CleanWEC_Test_constituency_trees
    python src/preprocessing/auto_mention_extraction.py --mention_path wiki_gold_mentions_json/clean_WEC/CleanWEC_Train_Event_gold_mentions.json --output_path wiki_gold_mentions_json/constituency_tree/clean_WEC/CleanWEC_Train_constituency_trees
    
    
    python src/preprocessing/divide_large_mentions.py --mention_path wiki_gold_mentions_json/clean_WEC/CleanWEC_Dev_Event_gold_mentions.json --constitiency_tree_path wiki_gold_mentions_json/constituency_tree/clean_WEC/CleanWEC_Dev_constituency_trees --output_path wiki_gold_mentions_json/clean_WEC/Min_CleanWEC_Dev_Event_gold_mentions.json
    python src/preprocessing/divide_large_mentions.py --mention_path wiki_gold_mentions_json/clean_WEC/CleanWEC_Test_Event_gold_mentions.json --constitiency_tree_path wiki_gold_mentions_json/constituency_tree/clean_WEC/CleanWEC_Test_constituency_trees --output_path wiki_gold_mentions_json/clean_WEC/Min_CleanWEC_Test_Event_gold_mentions.json
    python src/preprocessing/divide_large_mentions.py --mention_path wiki_gold_mentions_json/clean_WEC/CleanWEC_Train_Event_gold_mentions.json --constitiency_tree_path wiki_gold_mentions_json/constituency_tree/clean_WEC/CleanWEC_Train_constituency_trees --output_path wiki_gold_mentions_json/clean_WEC/Min_CleanWEC_Train_Event_gold_mentions.json
    
    
    
    '''