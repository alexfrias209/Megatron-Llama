#!/bin/bash

export CUDA_LAUNCH_BLOCKING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

TOKENIZER_MODEL=/workspace/megatron/llamaData/hf_format/tokenizer.model
CHECKPOINT_PATH=/workspace/megatron/checkpoints
DATA_PATH=/workspace/megatron/llamaData/my-gpt2_content_document


GPT_ARGS="
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --num-layers 24 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --seq-length 1024 \
    --max-position-embeddings 4096 \
    --micro-batch-size 4 \
    --global-batch-size 8 \
    --lr 0.00015 \
    --train-iters 500000 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --fp16 \
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model $TOKENIZER_MODEL \
    --no-load-optim \
    --no-load-rng \
    --untie-embeddings-and-output-weights \
    --use-rotary-position-embeddings \
    --normalization RMSNorm \
    --no-position-embedding \
    --no-masked-softmax-fusion \
    --no-query-key-layer-scaling
"

DATA_ARGS="
    --data-path $DATA_PATH
"

OUTPUT_ARGS="
    --log-interval 100 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10
"

torchrun /workspace/megatron/pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \





