#!/bin/bash
source scripts/download.sh

# Prepare dirs
MODEL_DIR=models
LLAMA_DIR=$MODEL_DIR/llama-7b
mkdir -p $LLAMA_DIR

# Download the llama delta weights
download_all_urls $PWD/scripts/llama_urls.txt $LLAMA_DIR