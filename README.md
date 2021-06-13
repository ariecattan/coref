# Cross-Document Coreference Resolution

This repository contains code and models for end-to-end cross-document coreference resolution, 
as decribed in our papers: 

- [Cross-document Coreference Resolution over Predicted Mentions (Findings of ACL 2021)](https://arxiv.org/abs/2106.01210)
- [Realistic Evaluation Principles for Cross-document Coreference Resolution (*SEM 2021)](https://arxiv.org/pdf/2106.04192.pdf)
 
The models are trained on ECB+, but they can be used for any setting of multiple documents.


## Getting started

* Install python3 requirements `pip install -r requirements.txt` 


### Extract mentions and raw text from ECB+ 

Run the following script in order to extract the data from ECB+ dataset
 and build the gold conll files. 
The ECB+ corpus can be downloaded [here](http://www.newsreader-project.eu/results/data/the-ecb-corpus/).

```
python get_ecb_data.py --data_path path_to_data
```



## Training Instructions


The core of our model is the pairwise scorer between two spans, 
which indicates how likely two spans belong to the same cluster.


#### Training method

We present 3 ways to train this pairwise scorer:

1. Pipeline: first train a span scorer, then train the pairwise scorer using the same spans at each epoch. 
2. Continue: pre-train the span scorer, then train the pairwise scorer while keep training the span scorer.
3. End-to-end: train together both models from scratch.


In order to choose the training method, you need to set the value of the `training_method` in 
the `config_pairwise.json` to `pipeline`, `continue` or `e2e`. 
In our [paper](), we found the `continue` method to perform the best for event coreference 
and we apply it for entity and ALL as well.

 
#### What are the labels ?

In ECB+, the entity and event coreference clusters are annotated separately, 
making it possible to train a model only on event or entity coreference. 
Therefore, our model also allows to be trained on events, entity, or both.
You need to set the value of the `mention_type` in 
the ``config_pairwise.json`` (and `config_span_scorer.json`) 
to `events`, `entities` or `mixed` (corresponding to ALL in the paper).





#### Running the model
 
In both `pipeline` and `continue` methods, you need to first run 
the span scorer model 

```
python train_span_scorer --config configs/config_span_scorer.json
```

For the pairwise scorer, run the following script
```
python train_pairwise_scorer --config configs/config_pairwise.json
```


Some important parameters in `config_pairwise.json`:
* `max_mention_span`
* `top_k`: pruning coefficient
* `training_method`: (pipeline, continue, e2e)
* `subtopic`: (true, false) whether to train at the topic or subtopic level (ECB+ notions).


#### Tuning threshold for agglomerative clustering


The training above will save 10 models (one for each epoch) in the specified directory, 
while each model is composed of a span_repr, a span scorer and a pairwise scorer. 
In order to find the best model and the best threshold for the agglomerative clustering, 
you need to do an hyperparameter search on the 10 models + several values for threshold, 
evaluated on the dev set. To do that, please set the `config_clustering.json` (`split`: `dev`) 
and run the two following scripts:

```
python tuned_threshold.py --config configs/config_clustering.json

python run_scorer.py [path_of_directory_of_conll_files] [mention_type]
```


## Prediction

Given the trained pairwise scorer, the best `model_num` and the `threshold` 
from the above training and tuning, set the `config_clustering.json` (`split`: `test`)
and run the following script. 

```
python predict.py --config configs/config_clustering
```

(`model_path` corresponds to the directory in which you've stored the trained models)

An important configuration in the `config_clustering` is the `topic_level`. 
If you set `false` , you need to provide the path to the predicted topics in `predicted_topics_path` 
to produce conll files at the corpus level. 

## Evaluation

The output of the `predict.py` script is a file in the standard conll format. 
Then, it's straightforward to evaluate it with its corresponding 
gold conll file (created in the first step), 
using the official conll coreference scorer
that you can find 
[here](https://github.com/conll/reference-coreference-scorers) or the [coval](https://github.com/ns-moosavi/coval/) system (python implementation).

Make sure to use the gold files of the same evaluation level (topic or corpus) as the predictions. 


## Notes


* If you chose to train the pairwise with the end-to-end method, you don't need to provide a `span_repr_path` or a `span_scorer_path` in the
`config_pairwise.json`.  

* If you use this model with gold mentions, the span scorer is not relevant, you should ignore the training method.

* If you're interested in a newer but heavier model, check out our [cross-encoder model](https://github.com/ariecattan/cross_encoder/)



## Team 

* [Arie Cattan](https://ariecattan.github.io/)
* Alon Eirew
* [Gabriel Stanovsky](https://gabrielstanovsky.github.io/)
* [Mandar Joshi](https://homes.cs.washington.edu/~mandar90/)
* [Ido Dagan](https://u.cs.biu.ac.il/~dagan/) 

 