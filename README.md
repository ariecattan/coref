# Cross-Document Coreference Resolution

This repository contains code and models for cross-document coreference resolution. 
The models are trained on ECB+, but they can be used for any setting of multiple documents.
Our model is the current state-of-the-art on ECB+ on gold mentions 
and the first model which can work on raw texts as well.



## Getting started

* python 3.7.3
* pytorch 1.3.0
* transformers 2.8.0
* spacy 2.1.4
* tqdm

### Extract mentions and raw text from ECB+ 

Run the following script in order to extract the data from ECB+ dataset
 and build the gold conll files. 
The ECB+ corpus can be downloaded [here](http://www.newsreader-project.eu/results/data/the-ecb-corpus/).

* ``python get_ecb_data.py --data_path path_to_data``



## Training Instructions

The core of our model is the pairwise scorer between two spans, 
which indicates how likely two spans belong to the same cluster.
We present 3 ways to train this pairwise scorer:

1. Pipeline: first train a span scorer, then train the pairwise scorer. 
Unlike Ontonotes, ECB+ does include singleton annotation, so it's possible to train separately the span scorer model.
2.  Fine-tune: first train the span scorer, then train the pairwise scorer
while continue training the span scorer.
3. End-to-end: train together the both models like in the 
[e2e-coref](https://github.com/kentonl/e2e-coref) model (Lee et al., 2017).

In order to choose the training method, you need to set the value of the `training_method` in 
the `config_pairwise.json` to `pipeline`, `fine_tune` or `e2e`
 
In both pipeline and fine-tuning methods, you need to first run 
the span scorer model 

* ``python train_span_scorer --config_file configs/config_span_scorer.json``

For the pairwise scorer, set the method in the config file as below and run the command
* ``python train_pairwise_scorer --config_file configs/config_pairwise.json``



Notice that if you use this model with gold mentions, the span scorer
is not relevant, you should use the `e2e` method

## Prediction

``python predict.py``

## Notes

* Since event and entity mentions in ECB+ are annotated separately, 
you must specify (in the config file) for both training and testing if you want to use the model for entity
or event coreference.

* If you chose to train with the end-to-end method, you don't need to provide a `span_repr_path` or a `span_scorer_path` in the
config file.  

* Notice that if you use this model with gold mentions, the span scorer is not relevant, you should use the e2e method

