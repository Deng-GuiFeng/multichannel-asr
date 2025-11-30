#!/bin/bash
# Example script for running the Multichannel ASR pipeline
# Modify the parameters according to your setup

python src/main.py \
    --model_id "openai/whisper-large-v3-turbo" \
    --device "cuda:0" \
    --src_dir "./data/input" \
    --tgt_dir "./data/output" \

