export LC_ALL=C.UTF-8
set -x

HOME=data/opus_100
DATA_DIR=data/opus_100/data-bin
USER_DIR=../fairseq_user_dir

negative_loss_lambda=1.0
translation_loss_lambda=1.0
continue_step=${1:-10000}
opus_checkpoints=$OPUS_HOME/checkpoints
exp_name=opus_baseline

lang_pairs="es-en,en-es,fr-en,en-fr,ro-en,en-ro,nl-en,en-nl,cs-en,en-cs,el-en,en-el,hu-en,en-hu,pl-en,en-pl,tr-en,en-tr,pt-en,en-pt,bg-en,en-bg,it-en,en-it,fi-en,en-fi,hr-en,en-hr,ar-en,en-ar,sr-en,en-sr,he-en,en-he,de-en,en-de,sl-en,en-sl,ru-en,en-ru,sv-en,en-sv,da-en,en-da,et-en,en-et,bs-en,en-bs,sk-en,en-sk,id-en,en-id,no-en,en-no,fa-en,en-fa,lt-en,en-lt,zh-en,en-zh,lv-en,en-lv,mk-en,en-mk,vi-en,en-vi,th-en,en-th,ja-en,en-ja,sq-en,en-sq,ms-en,en-ms,is-en,en-is,ko-en,en-ko,uk-en,en-uk,ca-en,en-ca,eu-en,en-eu,mt-en,en-mt,gl-en,en-gl,ml-en,en-ml,bn-en,en-bn,pa-en,en-pa,hi-en,en-hi,ta-en,en-ta,si-en,en-si,nb-en,en-nb,nn-en,en-nn,te-en,en-te,gu-en,en-gu,mr-en,en-mr,ne-en,en-ne,kn-en,en-kn,or-en,en-or,as-en,en-as,ka-en,en-ka,be-en,en-be,eo-en,en-eo,cy-en,en-cy,ga-en,en-ga,ug-en,en-ug,az-en,en-az,xh-en,en-xh,af-en,en-af,oc-en,en-oc,br-en,en-br,rw-en,en-rw,km-en,en-km,ku-en,en-ku,wa-en,en-wa,mg-en,en-mg,kk-en,en-kk,tg-en,en-tg,am-en,en-am,ps-en,en-ps,my-en,en-my,uz-en,en-uz,ur-en,en-ur,ky-en,en-ky,gd-en,en-gd,sh-en,en-sh,li-en,en-li,zu-en,en-zu,fy-en,en-fy,tk-en,en-tk,yi-en,en-yi,tt-en,en-tt,se-en,en-se,ha-en,en-ha,ig-en,en-ig"
lang_pair_list=(es-en en-es fr-en en-fr ro-en en-ro nl-en en-nl cs-en en-cs el-en en-el hu-en en-hu pl-en en-pl tr-en en-tr pt-en en-pt bg-en en-bg it-en en-it fi-en en-fi hr-en en-hr ar-en en-ar sr-en en-sr he-en en-he de-en en-de sl-en en-sl ru-en en-ru sv-en en-sv da-en en-da et-en en-et bs-en en-bs sk-en en-sk id-en en-id no-en en-no fa-en en-fa lt-en en-lt zh-en en-zh lv-en en-lv mk-en en-mk vi-en en-vi th-en en-th ja-en en-ja sq-en en-sq ms-en en-ms is-en en-is ko-en en-ko uk-en en-uk ca-en en-ca eu-en en-eu mt-en en-mt gl-en en-gl ml-en en-ml bn-en en-bn pa-en en-pa hi-en en-hi ta-en en-ta si-en en-si nb-en en-nb nn-en en-nn te-en en-te gu-en en-gu mr-en en-mr ne-en en-ne kn-en en-kn or-en en-or as-en en-as ka-en en-ka be-en en-be eo-en en-eo cy-en en-cy ga-en en-ga ug-en en-ug az-en en-az xh-en en-xh af-en en-af oc-en en-oc br-en en-br rw-en en-rw km-en en-km ku-en en-ku wa-en en-wa mg-en en-mg kk-en en-kk tg-en en-tg am-en en-am ps-en en-ps my-en en-my uz-en en-uz ur-en en-ur ky-en en-ky gd-en en-gd sh-en en-sh li-en en-li zu-en en-zu fy-en en-fy tk-en en-tk yi-en en-yi tt-en en-tt se-en en-se ha-en en-ha ig-en en-ig)
pairs_num=187

warm_up=1
lr=0.00005
save_dir=${opus_checkpoints}/$exp_name/continue_train.continue_step${continue_step}.negative_loss_lambda${negative_loss_lambda}.translation_loss_lambda${translation_loss_lambda}.lr${lr}.wu${warm_up}
mkdir -p $save_dir
baseline_model=${opus_checkpoints}/$exp_name/checkpoint_best.pt
# 4*gpu
CUDA_VISIBLE_DEVICES=4,5,6,7 fairseq-train $DATA_DIR --arch transformer_vaswani_wmt_en_de_big --share-all-embeddings \
    --user-dir $USER_DIR \
    --task translation_multi_simple_epoch_negative_loss --sampling-method "temperature" \
    --sampling-temperature 5 --encoder-langtok src --decoder-langtok \
    --langs "af,am,ar,as,az,be,bg,bn,br,bs,ca,cs,cy,da,de,el,en,eo,es,et,eu,fa,fi,fr,fy,ga,gd,gl,gu,ha,he,hi,hr,hu,id,ig,is,it,ja,ka,kk,km,kn,ko,ku,ky,li,lt,lv,mg,mk,ml,mr,ms,mt,my,nb,ne,nl,nn,no,oc,or,pa,pl,ps,pt,ro,ru,rw,se,sh,si,sk,sl,sq,sr,sv,ta,te,tg,th,tk,tr,tt,ug,uk,ur,uz,vi,wa,xh,yi,zh,zu" \
    --lang-pairs $lang_pairs \
    --criterion label_smoothed_cross_entropy_with_negative_tokens \
    --translation-loss-lambda ${translation_loss_lambda} --negative-loss-lambda ${negative_loss_lambda} \
    --label-smoothing 0.1 \
    --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
    --lr-scheduler inverse_sqrt --lr ${lr} --warmup-updates ${warm_up} --max-update ${continue_step} \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.0 --max-tokens 4096 --update-freq 2 \
    --save-interval-updates 500 --keep-interval-updates 50 --no-epoch-checkpoints \
    --log-format simple --log-interval 2 --seed 1234 --fp16 --ddp-backend no_c10d \
    --save-dir $save_dir \
    --skip-invalid-size-inputs-valid-test \
    --tensorboard-logdir ${save_dir}/tensorboard \
    --restore-file $baseline_model \
    --reset-optimizer --reset-dataloader --reset-lr-scheduler --reset-meters | tee -a $save_dir/train.log

