#!/bin/bash
set -x

HOME=/data/iwslt_4
DATA_DIR=/data/iwslt_4/data-bin





dropout=0.3
vocab_size=40000
warmup=8000
checkpoints=$HOME/checkpoints
exp_name=iwslt4_baseline
lang_pairs="en-nl,en-ro,en-it,nl-en,it-en,ro-en"
lang_dict=$DATA_DIR/lang_dict.txt

# 4*gpus
CUDA_VISIBLE_DEVICES=0,1,2,3 fairseq-train $DATA_DIR \
    --task translation_multi_simple_epoch \
    --arch transformer \
    --encoder-layers 5 --decoder-layers 5 \
    --sampling-method "concat" \
    --decoder-langtok --encoder-langtok src \
    --lang-dict ${lang_dict} \
    --lang-pairs ${lang_pairs} \
    --source-dict $DATA_DIR/dict.txt \
    --target-dict $DATA_DIR/dict.txt \
    --share-all-embeddings \
    --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --clip-norm 0.0 \
    --lr-scheduler inverse_sqrt \
    --warmup-init-lr 1e-07 \
    --warmup-updates $warmup \
    --lr 0.0005 \
    --stop-min-lr 1e-09 \
    --dropout $dropout \
    --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --max-tokens 4000 --seed 1 --ddp-backend no_c10d \
    --max-update 100000 --log-interval 200 \
    --save-dir $checkpoints/$exp_name --tensorboard-logdir $checkpoints/$exp_name/tensorboard \
    --fp16 --fp16-no-flatten-grads | tee -a $checkpoints/$exp_name/train.log 
