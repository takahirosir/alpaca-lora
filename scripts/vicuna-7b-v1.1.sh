#!/bin/bash
source scripts/download.sh

# Prepare dirs
MODEL_DIR=models
VICUNA_DIR=$MODEL_DIR/vicuna-7b-v1.1
mkdir -p $VICUNA_DIR

# Download the vicuna delta weights
download_all_urls $PWD/scripts/vicuna-7b-v1.1.txt $VICUNA_DIR
