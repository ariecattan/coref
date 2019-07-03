import argparse
import os, fnmatch
import xml.etree.ElementTree as ET

parser = argparse.ArgumentParser()
parser.add_argument('--path_to_data')
parser.add_argument('--output_path')
args = parser.parse_args()


def get_raw_data_from_file(doc_path):
    data = {}
    tree = ET.parse(doc_path)
    root = tree.getroot()

    for child in root:
        if child.tag == 'token':
            sentence = child.attrib['sentence']
            data[sentence] = [] if sentence not in data else data[sentence]
            data[sentence].append(child.text)


    return data



def main():
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    pattern = '*.xml'
    for folder in os.listdir(args.path_to_data):
        print('Processing folder {}'.format(folder))
        with open(os.path.join(args.output_path, folder), 'w') as f:
            for doc in os.listdir(os.path.join(args.path_to_data, folder)):
                 if fnmatch.fnmatch(doc, pattern):
                    documents = get_raw_data_from_file(os.path.join(args.path_to_data, folder, doc))
                    f.write(doc + '\n')
                    for sentence, tokens in documents.items():
                        f.write(sentence + ' ' + ' '.join(tokens) + '\n')
                    f.write('\n')




if __name__ == '__main__':
    main()


