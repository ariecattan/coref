#!/usr/bin/env bash
DIRECTORY=~/coref/data/ecb/docs/*.xml


for f in $DIRECTORY
    do echo "Processing $f"
    cd ~/mandar_coref
    bs=`basename "$f"`
    bs=${bs%.*}
    GPU=0 python predict.py spanbert_large $f ~/coref/data/ecb/wd_coref/$bs.json
done
