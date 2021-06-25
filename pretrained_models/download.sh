#!/usr/bin/env bash
############################################################## 
# This script is used to download resources for ADERel experiments
############################################################## 

cd "pretrained_models"
mkdir "biobert"
wget https://huggingface.co/monologg/biobert_v1.1_pubmed/resolve/main/config.json -O "biobert/config.json"
wget https://huggingface.co/monologg/biobert_v1.1_pubmed/resolve/main/pytorch_model.bin -O "biobert/pytorch_model.bin"
wget https://huggingface.co/monologg/biobert_v1.1_pubmed/resolve/main/vocab.txt -O "biobert/vocab.txt"

cd "pretrained_models"
mkdir "scibert"
wget https://huggingface.co/allenai/scibert_scivocab_cased/resolve/main/pytorch_model.bin -O "scibert/pytorch_model.bin"
wget https://huggingface.co/allenai/scibert_scivocab_cased/resolve/main/config.json -O "scibert/config.json"
wget https://huggingface.co/allenai/scibert_scivocab_cased/resolve/main/vocab.txt -O "scibert/vocab.txt"

cd "pretrained_models"
mkdir "bert"
wget https://huggingface.co/bert-base-cased/resolve/main/pytorch_model.bin -O "bert/pytorch_model.bin"
wget https://huggingface.co/bert-base-cased/resolve/main/config.json -O "bert/config.json"
wget https://huggingface.co/bert-base-cased/resolve/main/vocab.txt -O "bert/vocab.txt"

