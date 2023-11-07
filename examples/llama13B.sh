 #!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1

TOKENIZER_MODEL=/workspace/megatron/llamaData/tokenizerFiles/tokenizer.model
CHECKPOINT_PATH=/workspace/megatron/checkpoints/llama13b
DATA_PATH=/workspace/megatron/llamaData/my-llama_content_document #If using different json key then name will change

#Change Global Batch-Size so that it's trained with a global batch-size of 4M tokens
GPT_ARGS="
    --tensor-model-parallel-size 2 \
    --pipeline-model-parallel-size 1 \
    --num-layers 40 \
    --hidden-size 5120 \
    --num-attention-heads 40 \
    --seq-length 4096  \
    --max-position-embeddings 4096 \
    --micro-batch-size 4 \
    --global-batch-size 8 \
    --lr 3.0e-4 \
    --train-iters 500000 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --fp16 \
    --adam-beta1 0.9 \
	--adam-beta2 0.95 \
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model $TOKENIZER_MODEL \
    --no-load-optim \
    --no-load-rng \
    --swiglu \
    --untie-embeddings-and-output-weights \
    --use-rotary-position-embeddings \
    --normalization RMSNorm \
    --no-position-embedding \
    --lr-warmup-iters 2000 \
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
