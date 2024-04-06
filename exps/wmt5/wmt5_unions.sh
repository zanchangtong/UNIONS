export LC_ALL=C.UTF-8
set -x

HOME=/data/wmt5
DATA_DIR=/data/wmt5/data-bin
USER_DIR=./fairseq_user_dir


checkpoints=$HOME/checkpoints

lang_pairs="zh-en,de-en,fr-en,ro-en,en-zh,en-de,en-fr,en-ro"
lang_pair_list=(zh-en de-en fr-en ro-en en-zh en-de en-fr en-ro)
pairs_num=8
lr=0.00007
step=500
warm_up=1
negative_loss_lambda=1.0
translation_loss_lambda=1.0
exp_name=wmt5_baseline

save_dir=$checkpoints/$exp_name/continue_step${step}.negative_loss_lambda${negative_loss_lambda}.translation_loss_lambda${translation_loss_lambda}.lr${lr}.wu${warm_up}
mkdir -p $save_dir
baseline_model=${checkpoints}/$exp_name/checkpoint_best.pt
# 4*gpus
CUDA_VISIBLE_DEVICES=4,5,6,7 fairseq-train $DATA_DIR \
    --restore-file ${baseline_model} \
    --arch transformer_vaswani_wmt_en_de_big --share-all-embeddings \
    --user-dir $USER_DIR \
    --task translation_multi_simple_epoch_negative_loss --sampling-method "temperature" \
    --sampling-temperature 5 --encoder-langtok src --decoder-langtok \
    --langs "de,en,fr,ro,zh" \
    --lang-pairs $lang_pairs \
    --criterion label_smoothed_cross_entropy_with_negative_tokens \
    --translation-loss-lambda ${translation_loss_lambda} --negative-loss-lambda ${negative_loss_lambda} \
    --label-smoothing 0.1 \
    --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
    --lr-scheduler inverse_sqrt --lr ${lr} --warmup-updates ${warm_up} --max-update ${step} \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.0 --max-tokens 8192 --update-freq 1 \
    --save-interval-updates 50 --keep-interval-updates 50 --no-epoch-checkpoints \
    --log-format simple --log-interval 2 --seed 1234 --fp16 --ddp-backend no_c10d \
    --save-dir $save_dir \
    --skip-invalid-size-inputs-valid-test \
    --reset-optimizer --reset-dataloader --reset-lr-scheduler --reset-meters \
    --tensorboard-logdir ${save_dir}/tensorboard | tee -a $save_dir/train.log

