#!/bin/bash

GPUIDS="${1:-0}"
DATA_DIR="${2:-data/logic_graph}"
OUT_DIR="${3:-models/graph/origin}"
BATCH_SIZE="${4:-32}"
NUM_LAYERS="${5:-1}"

mkdir -p ${OUT_DIR}

python train.py \
    -activation 'relu' \
    -highway 'tanh' \
    -n_gcn_layer 2 \
    -gcn_edge_dropout 0.1 \
    -gcn_dropout 0.1 \
    -data_type logic \
    -data ${DATA_DIR}/logic \
    -save_model ${OUT_DIR}/logic-model \
    -layers ${NUM_LAYERS} \
    -report_every 50 \
    -train_steps 10000 \
    -valid_steps 1000 \
    -rnn_size 800 \
    -word_vec_size 800 \
    -gcn_vec_size 800 \
    -encoder_type gcn \
    -decoder_type rnn \
    -batch_size ${BATCH_SIZE} \
    -max_generator_batches 50 \
    -save_checkpoint_steps 2500 \
    -decay_steps 150 \
    -optim sgd \
    -max_grad_norm 3 \
    -learning_rate_decay 0.8 \
    -start_decay_steps 6000 \
    -learning_rate 0.5 \
    -dropout 0.5 \
    -gpu_ranks ${GPUIDS} \
    -seed 123 
    # -copy_attn