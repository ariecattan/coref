import argparse
import spacy
import neuralcoref
import numpy as np


nlp = spacy.load('en')
neuralcoref.add_to_pipe(nlp)



def read_file(doc):
    docs = {}
    with open(doc, 'r') as f:
        for line in f:
            data = line.split('\t')
            if len(data) == 4
                doc_name, _, _, tok_text = data
                docs[doc_name] = [] if doc_name not in docs else docs[doc_name]
                docs[doc_name].append(tok_text)

    return docs


def spacy_coref_doc(tokens):
    doc = u' '.join(tokens)
    doc = nlp(doc)

    return doc._.coref_clusters


def main(args):
    docs = read_file(args.data)
    clusters = [spacy_coref_doc(doc) for doc in docs]



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='conll_file', default='ecb_data/dev_text.txt')

    args = parser.parse_args()
    main(args)


