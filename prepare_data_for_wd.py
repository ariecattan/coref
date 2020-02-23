import jsonlines
import argparse
from transformers import BertTokenizer
import json
import os
from utils import *


parser = argparse.ArgumentParser()
parser.add_argument('--data_folder', type=str, default='data/ecb/mentions')
parser.add_argument('--output_dir', type=str, default='data/ecb/docs')
args = parser.parse_args()


if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-large-cased')
    sp_tokens = []

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)



    with jsonlines.open(os.path.join(args.output_dir, 'all_docs.jsonl'), 'w') as f_w:
        for i, sp in enumerate(['train', 'dev', 'test']):
            with open(os.path.join(args.data_folder, sp + '.json'), 'r') as f:
                data = json.load(f)
            # sp_tokens.append(data)

            for doc, tokens in data.items():
                print('Processing doc {}'.format(doc))
                doc_bert_tokens = []
                doc_alignment = []
                doc_map_sentence = []
                for i, (sentence_num, token_id, token, _, _) in enumerate(tokens):
                    bert_tokens = tokenizer.tokenize(token)
                    if bert_tokens:
                        doc_bert_tokens.extend(bert_tokens)
                        doc_map_sentence.extend([sentence_num] * len(bert_tokens))
                        doc_alignment.extend([i] * len(bert_tokens))

                segments = split_doc_into_segments(doc_bert_tokens, doc_map_sentence, with_special_tokens=True)


                bert_sentences, bert_sentence_map, bert_alignment, bert_speakers = [], [], [], []
                for start, end in zip(segments, segments[1:]):
                    bert_sentences.append(["[CLS]"] + doc_bert_tokens[start:end] + ["[SEP]"])
                    bert_speakers.append(["[SPL]"] + ['-'] * (end-start) + ["[SPL]"])
                    bert_sentence_map.extend([doc_map_sentence[start]] + doc_map_sentence[start:end] + [doc_map_sentence[end-1]])
                    bert_alignment.extend([doc_alignment[start]] + doc_alignment[start:end] + [doc_alignment[end-1]])


                f_w.write({
                    "clusters": [],
                    "doc_key": "nw",
                    "doc_name": doc,
                    "sentences": bert_sentences,
                    "speakers": bert_speakers,
                    "sentence_map": bert_sentence_map,
                    "subtoken_map": bert_alignment
                })





                    # {
                    #     "clusters": [],  # leave this blank
                    #     "doc_key": "nw",
                    # # key closest to your domain. "nw" is newswire. See the OntoNotes documentation.
                    #     "sentences": [["[CLS]", "subword1", "##subword1", ".", "[SEP]"]],
                    # # list of BERT tokenized segments. Each segment should be less than the max_segment_len in your config
                    #     "speakers": [["[SPL]", "-", "-", "-", "[SPL]"]],
                    # # speaker information for each subword in sentences
                    #     "sentence_map": [0, 0, 0, 0, 0],
                    # # flat list where each element is the sentence index of the subwords
                    #     "subtoken_map": [0, 0, 0, 1, 1]
                    # # flat list containing original word index for each subword. [CLS]  and the first word share the same index
                    # }

