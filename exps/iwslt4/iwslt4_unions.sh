#!/bin/bash
set -x

HOME=/data/iwslt_4
DATA_DIR=/data/iwslt_4/data-bin
USER_DIR=./fairseq_user_dir


checkpoints=$HOME/checkpoints
exp_name=iwslt4_baseline
dropout=0.3
vocab_size=40000
warmup=1
negative_loss_lambda=1.0
translation_loss_lambda=0.0
gpus=${1:-"0"}
continue_step=500
lr=0.00005
mt=1000

lang_pairs="en-nl,en-ro,en-it,nl-en,it-en,ro-en"
lang_dict=$DATA_DIR/lang_dict.txt
save_dir=${checkpoints}/$exp_name/continue_train.continue_step${continue_step}.negative_loss_lambda${negative_loss_lambda}.translation_loss_lambda${translation_loss_lambda}.lr${lr}.wu${warmup}.mt${mt}

mkdir -p $save_dir
baseline_model=${checkpoints}/$exp_name/checkpoint_best.pt
# 1*gpus
CUDA_VISIBLE_DEVICES=$gpus fairseq-train $DATA_DIR \
    --user-dir $USER_DIR \
    --task translation_multi_simple_epoch_negative_loss \
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
    --lr ${lr} \
    --stop-min-lr 1e-09 \
    --dropout $dropout \
    --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy_with_negative_tokens \
    --label-smoothing 0.1 --negative-loss-lambda ${negative_loss_lambda} --translation-loss-lambda ${translation_loss_lambda} \
    --max-tokens ${mt} --seed 1 --ddp-backend no_c10d \
    --max-update ${continue_step} --log-interval 2 \
    --save-dir $save_dir --tensorboard-logdir $save_dir/tensorboard \
    --restore-file $baseline_model \
    --reset-optimizer --reset-dataloader --reset-lr-scheduler --reset-meters \
    --fp16 --fp16-no-flatten-grads \
    --save-interval-updates 50 --keep-interval-updates 50 --no-epoch-checkpoints | tee -a $save_dir/train.log 

