#!/bin/bash
source scripts/download.sh

# Prepare dirs
MODEL_DIR=models
LLAMA_DIR=$MODEL_DIR/llama
VICUNA_DIR=$MODEL_DIR/vicuna-7b-delta-v0
MINIGPT4_DIR=$MODEL_DIR/minigpt4
OUTPUT_DIR=$MODEL_DIR/output
mkdir -p $LLAMA_DIR
mkdir -p $OUTPUT_DIR

# Download the original llama model weights
download_all_urls $PWD/scripts/llama_urls.txt $LLAMA_DIR

# Download the vicuna delta weights
download_all_urls $PWD/scripts/vicuna_urls.txt $VICUNA_DIR

# Generate vicuna weights
git submodule update --init
pip install -e FastChat
python -m fastchat.model.apply_delta --base-model-path $LLAMA_DIR  --target-model-path $OUTPUT_DIR  --delta-path $VICUNA_DIR
