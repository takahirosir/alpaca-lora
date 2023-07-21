#!/bin/bash
source scripts/download.sh

# Prepare dirs
MODEL_DIR=models
ALPACA_DIR=$MODEL_DIR/alpaca-lora-7b
mkdir -p $ALPACA_DIR

# Download the alpaca-lora-7b delta weights
download_all_urls $PWD/scripts/alpaca-lora-7b_urls.txt $ALPACA_DIR