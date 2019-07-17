# Coref

## Pre-processing 

### Get Mentions and Coref chains

``python get_ecb_data.py ``

``python get_meantime_data.py ``

### Prepare raw data files

``python src/preprocessing/extract_data_from_xml.py --path_to_data datasets/ECB+_LREC2014/ECB+ --output_path data/ecb/ecb_raw_data``

``python src/preprocessing/extract_data_from_xml.py --path_to_data datasets/meantime_newsreader_english_oct15/intra_cross-doc_annotation/ --output_path data/meantime/meantime_raw_data``

### Run constituency tree model for all sentences

``python src/preprocessing/auto_mention_extraction.py  --raw_data data/ecb/ecb_raw_data --output_path data/ecb/ecb_constituency_tree``

``python src/preprocessing/auto_mention_extraction.py  --raw_data data/meantime/meantime_raw_data --output_path data/meantime/meantime_constituency_tree``

