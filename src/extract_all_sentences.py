from utils import arr_to_text
from read_corpus import read_folder
import sys

topic_path = sys.argv[1]


def main():
    topic = read_folder(topic_path)
    for subtopic, docs in topic.items():
        for file, doc_id, sentences in docs:
            print('File: {}'.format(file))
            sents = arr_to_text(sentences)
            for sent in sents:
                print(sent)
            print()


if __name__ == '__main__':
    main()