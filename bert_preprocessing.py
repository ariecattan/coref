import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, RobertaModel, RobertaTokenizer
import argparse
import json
import pickle
import os
import logging




parser = argparse.ArgumentParser()
parser.add_argument('--transformer_model', type=str, default='bert-base-cased')
parser.add_argument('--spanbert_path', type=str, default='')
parser.add_argument('--data_path', type=str, default='data/ecb')
parser.add_argument('--output_path', type=str, default='data/ecb/bert_docs')
parser.add_argument('--segment_length', type=int, default=512)
parser.add_argument('--gpu', type=int, default=1)
args = parser.parse_args()

format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
logging.basicConfig(format=format, level=logging.INFO,
                    filename=os.path.join(args.output_path, 'log.txt'),
                    filemode='w', datefmt='%Y-%m-%d %H:%M:%S')
logging.getLogger().addHandler(logging.StreamHandler())
logger = logging.getLogger(__name__)





def make_segments_with_padding(doc):
    segments = []
    current_token = 0
    while current_token < len(doc['bert_tokens']):
        end_token = min(len(doc['bert_tokens']) - 1, current_token + args.segment_length - 1)
        sentence_end = doc['sentence_alignment'][end_token]
        if end_token != len(doc['bert_tokens']) - 1 and doc['sentence_alignment'][end_token + 1] == sentence_end:
            while end_token >= current_token and doc['sentence_alignment'][end_token] == sentence_end:
                end_token -= 1

            if end_token < current_token:
                raise ValueError(doc['bert_tokens'])

        segment = doc['bert_tokens'][current_token:end_token+1]
        segments.append(segment)
        current_token = end_token + 1

    return segments



def get_token_ids(segments, tokenizer):
    ids, masks = [], []
    for segment in segments:
        input_ids = tokenizer.convert_tokens_to_ids(segment)
        input_mask = [1] * len(input_ids) + [0] * (args.segment_length - len(input_ids))
        input_ids += [0] * (args.segment_length - len(input_ids))

        ids.append(input_ids)
        masks.append(input_mask)

    return ids, masks



def get_bert_embedding(docs, bert_model, bert_tokenizer):
    bert_embeddings = {}
    num_of_multiple_segments = 0
    for doc_id, doc in docs.items():
        print('Processing doc: {}'.format(doc_id))
        segments = make_segments_with_padding(doc)
        if len(segments) > 1:
            num_of_multiple_segments += 1
            print('Doc {} was splitted into more than one segment'.format(doc_id))
        input_ids, input_masks = get_token_ids(segments, bert_tokenizer)

        with torch.no_grad():
            bert_output, _ = bert_model(torch.tensor(input_ids).to(device),  attention_mask=torch.tensor(input_masks).to(device))
            shape = bert_output.shape
            bert_output = bert_output.view(shape[0] * shape[1], shape[2])
            masks = torch.tensor([x for segment in input_masks for x in segment])
            bert_output = bert_output[torch.nonzero(masks).squeeze(1)]
            bert_embeddings[doc_id] = bert_output.detach()

    print('{} have more than one segment'.format(num_of_multiple_segments))
    return bert_embeddings





if __name__ == '__main__':
    device = 'cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu'
    print('Device: {}'.format(device))

    bert_tokenizer = BertTokenizer.from_pretrained(args.transformer_model)

    if args.spanbert_path:
        bert_model = BertModel.from_pretrained(args.spanbert_path).to(device)
        args.transformer_model = args.spanbert_path.split('/')[-1]
    else:
        bert_model = BertModel.from_pretrained(args.transformer_model).to(device)

    all_docs = {}
    for data in ['train', 'dev', 'test']:
        with open(os.path.join(args.data_path, 'mentions', data + '.json'), 'r') as f:
            all_docs.update(json.load(f))


    docs_bert_embeddings = get_bert_embedding(all_docs, bert_model, bert_tokenizer)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)



    with open(os.path.join(args.output_path, args.transformer_model + '_' + str(args.segment_length)), 'wb') as f:
        pickle.dump(docs_bert_embeddings, f)