#!/bin/bash

ROOT="<your code root>"

LLM_PATH=$1

VLLM_DIR=$ROOT/vllm-0.2.7

echo "Start vllm server on path $LLM_PATH !!!"

if [[ $1 == *"alpaca"* ]]; then
    python $VLLM_DIR/vllm/entrypoints/openai/api_server.py --model $LLM_PATH --tensor-parallel-size 2 --dtype half --max-num-batched-tokens 32768 --chat-template $VLLM_DIR/examples/template_alpaca.jinja
else
    # mistral and llama has
    python $VLLM_DIR/vllm/entrypoints/openai/api_server.py --model $LLM_PATH --tensor-parallel-size 2 --dtype half --max-num-batched-tokens 32768
fi