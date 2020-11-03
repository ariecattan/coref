# Cross-Document Coreference Resolution

This repository contains code and models for end-to-end cross-document coreference resolution, as decribed in our paper: [Streamlining Cross-Document Coreference Resolution: Evaluation and Modeling](https://arxiv.org/abs/2009.11032) 
The models are trained on ECB+, but they can be used for any setting of multiple documents.



```
    @article{Cattan2020StreamliningCC,
      title={Streamlining Cross-Document Coreference Resolution: Evaluation and Modeling},
      author={Arie Cattan and Alon Eirew and Gabriel Stanovsky and Mandar Joshi and I. Dagan},
      journal={ArXiv},
      year={2020},
      volume={abs/2009.11032}
    }
```


## Getting started

* Install python3 requirements `pip install -r requirements.txt` 


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
2.  Continue: first train the span scorer, then train the pairwise scorer
while continue training the span scorer.
3. End-to-end: train together the both models.

In order to choose the training method, you need to set the value of the `training_method` in 
the `config_pairwise.json` to `pipeline`, `continue` or `e2e`

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

An important configuration in the `config_clustering` is the `topic_level`. 
If you set `false`, you need to provide the path to the predicted topics in `predicted_topics_path` 
to produce conll files at the corpus level. 

## Evaluation

The output of the `predict.py` script is a file in the standard conll format. 
Then, it's straightforward to evaluate it with its corresponding 
gold conll file (created in the first step), 
using the official conll coreference scorer
that you can find 
[here](https://github.com/conll/reference-coreference-scorers).

Make sure to use the gold files of the same evaluation level (topic or corpus) as the predictions. 


## Notes


* If you chose to train with the end-to-end method, you don't need to provide a `span_repr_path` or a `span_scorer_path` in the
config file.  

* Notice that if you use this model with gold mentions, 
the span scorer is not relevant, you should ignore the training
method.

* If you're interested in a newer model, check out our [cross-encoder model](https://github.com/ariecattan/cross_encoder/)