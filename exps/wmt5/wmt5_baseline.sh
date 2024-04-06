export LC_ALL=C.UTF-8
set -x

HOME=/data/wmt5
DATA_DIR=/data/iwslt_4/data-bin



checkpoints=$HOME/checkpoints
lang_pairs="zh-en,de-en,fr-en,ro-en,en-zh,en-de,en-fr,en-ro"
lang_pair_list=(zh-en de-en fr-en ro-en en-zh en-de en-fr en-ro)
pairs_num=8
lr=0.0007
exp_name=wmt5_baseline

mkdir -p $checkpoints/$exp_name
# 8*gpus
fairseq-train $DATA_DIR --arch transformer_vaswani_wmt_en_de_big --share-all-embeddings \
    --task translation_multi_simple_epoch --sampling-method "temperature" \
    --sampling-temperature 5 --encoder-langtok src --decoder-langtok \
    --langs "de,en,fr,ro,zh" \
    --lang-pairs $lang_pairs \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
    --lr-scheduler inverse_sqrt --lr ${lr} --warmup-updates 4000 --max-update 100000 \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.0 --max-tokens 8192 --update-freq 8 \
    --save-interval-updates 1000 --keep-interval-updates 20 --no-epoch-checkpoints \
    --log-format simple --log-interval 100 --seed 1234 --fp16 --ddp-backend no_c10d \
    --save-dir $checkpoints/$exp_name --max-source-positions 1024 --max-target-positions 1024 \
    --skip-invalid-size-inputs-valid-test \
    --tensorboard-logdir ${checkpoints}/$exp_name/tensorboard | tee -a $checkpoints/$exp_name/train.log

