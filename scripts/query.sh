#!/bin/bash

SCRIPT_DIR=$(dirname "$0")
ROOT=$SCRIPT/../

export PYTHONPATH=$ROOT:$PYTHONPATH

MODEL=${1:-"meta-llama/Llama-2-7b-hf"}

model_base=$(basename "$MODEL")

# Start the server
nohup $SCRIPT_DIR/server.sh $MODEL > $SCRIPT_DIR/server_$model_base.log 2>&1 &
server_pid=$!

# Wait for the server to start
echo "Waiting for the server to start..."
sleep 150

# Check if the server is running
if ps -p $server_pid > /dev/null; then
    echo "Server started successfully!"
else
    echo "Failed to start the server"
    exit 1
fi

DATA=$ROOT/data/
SCRIPT=$ROOT/src/

RESULTS=$ROOT/email_results/
mkdir -p $RESULTS

# make queries
DATA_FILE=$DATA/email/multi/email.jsonl
DATA_KEY="email-multi"

echo "$model_base"

python $SCRIPT/test_feedback_main_email_multi.py main --feedback_file $DATA_FILE --output_file $RESULTS/$model_base.$DATA_KEY.jsonl --model_name $MODEL