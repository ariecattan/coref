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


#### Training method

We present 3 ways to train this pairwise scorer:

1. Pipeline: first train a span scorer, then train the pairwise scorer. 
Unlike Ontonotes, ECB+ does include singleton annotation, so it's possible to train separately the span scorer model.
2.  Fine-tune: first train the span scorer, then train the pairwise scorer
while continue training the span scorer.
3. End-to-end: train together the both models.

In order to choose the training method, you need to set the value of the `training_method` in 
the `config_pairwise.json` to `pipeline`, `fine_tune` or `e2e`

In our experiments, we found the `e2e` method to perform the best for event coreference.

 
#### What are the labels ?

In ECB+, the entity and event coreference clusters are annotated separately, 
making it possible to train a model only on event or entity coreference. 
Therefore, our model also allows to be trained on events, entity, or both.
You need to set the value of the `mention_type` in 
the ``config_pairwise.json`` (and `config_span_scorer.json`) 
to `events`, `entities` or `mixed`.


#### Running the model
 
In both pipeline and fine-tuning methods, you need to first run 
the span scorer model 

* ``python train_span_scorer --config configs/config_span_scorer.json``

For the pairwise scorer, run the following script
* ``python train_pairwise_scorer --config configs/config_pairwise.json``



## Prediction

Given the pairwise scorer trained above, we use an agglomerative
clustering in order to cluster the candidate spans into coreference clusters. 


``python predict.py --config configs/config_clustering``

(`model_path` corresponds to the directory in which you've stored the trained models)

## Evaluation

The output of the `predict.py` script is a file with conll format. 
Then, it's straightforward to evaluate it with its corresponding 
gold conll file (created in the first step), 
using the official conll coreference scorer
that you can find 
[here](https://github.com/conll/reference-coreference-scorers).




## Notes


* If you chose to train with the end-to-end method, you don't need to provide a `span_repr_path` or a `span_scorer_path` in the
config file.  

* Notice that if you use this model with gold mentions, 
the span scorer is not relevant, you should ignore the training
method.

