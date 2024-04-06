export LC_ALL=C.UTF-8
set -x
# 8*GPUS
spm_decode=PATH_TO_SPM_DECODE
spm_model=PATH_TO_SPM_MODEL/spm_64k.model
OPUS_HOME=PATH_TO_SAVE_OUTPUT/opus_100
opus_checkpoints=$OPUS_HOME/checkpoints
DATA_DIR=DATA_HOME/opus_100

lang_pairs="es-en,en-es,fr-en,en-fr,ro-en,en-ro,nl-en,en-nl,cs-en,en-cs,el-en,en-el,hu-en,en-hu,pl-en,en-pl,tr-en,en-tr,pt-en,en-pt,bg-en,en-bg,it-en,en-it,fi-en,en-fi,hr-en,en-hr,ar-en,en-ar,sr-en,en-sr,he-en,en-he,de-en,en-de,sl-en,en-sl,ru-en,en-ru,sv-en,en-sv,da-en,en-da,et-en,en-et,bs-en,en-bs,sk-en,en-sk,id-en,en-id,no-en,en-no,fa-en,en-fa,lt-en,en-lt,zh-en,en-zh,lv-en,en-lv,mk-en,en-mk,vi-en,en-vi,th-en,en-th,ja-en,en-ja,sq-en,en-sq,ms-en,en-ms,is-en,en-is,ko-en,en-ko,uk-en,en-uk,ca-en,en-ca,eu-en,en-eu,mt-en,en-mt,gl-en,en-gl,ml-en,en-ml,bn-en,en-bn,pa-en,en-pa,hi-en,en-hi,ta-en,en-ta,si-en,en-si,nb-en,en-nb,nn-en,en-nn,te-en,en-te,gu-en,en-gu,mr-en,en-mr,ne-en,en-ne,kn-en,en-kn,or-en,en-or,as-en,en-as,ka-en,en-ka,be-en,en-be,eo-en,en-eo,cy-en,en-cy,ga-en,en-ga,ug-en,en-ug,az-en,en-az,xh-en,en-xh,af-en,en-af,oc-en,en-oc,br-en,en-br,rw-en,en-rw,km-en,en-km,ku-en,en-ku,wa-en,en-wa,mg-en,en-mg,kk-en,en-kk,tg-en,en-tg,am-en,en-am,ps-en,en-ps,my-en,en-my,uz-en,en-uz,ur-en,en-ur,ky-en,en-ky,gd-en,en-gd,sh-en,en-sh,li-en,en-li,zu-en,en-zu,fy-en,en-fy,tk-en,en-tk,yi-en,en-yi,tt-en,en-tt,se-en,en-se,ha-en,en-ha,ig-en,en-ig"
lang_pair_list=(es-en en-es fr-en en-fr ro-en en-ro nl-en en-nl cs-en en-cs el-en en-el hu-en en-hu pl-en en-pl tr-en en-tr pt-en en-pt bg-en en-bg it-en en-it fi-en en-fi hr-en en-hr ar-en en-ar sr-en en-sr he-en en-he de-en en-de sl-en en-sl ru-en en-ru sv-en en-sv da-en en-da et-en en-et bs-en en-bs sk-en en-sk id-en en-id no-en en-no fa-en en-fa lt-en en-lt zh-en en-zh lv-en en-lv mk-en en-mk vi-en en-vi th-en en-th ja-en en-ja sq-en en-sq ms-en en-ms is-en en-is ko-en en-ko uk-en en-uk ca-en en-ca eu-en en-eu mt-en en-mt gl-en en-gl ml-en en-ml bn-en en-bn pa-en en-pa hi-en en-hi ta-en en-ta si-en en-si nb-en en-nb nn-en en-nn te-en en-te gu-en en-gu mr-en en-mr ne-en en-ne kn-en en-kn or-en en-or as-en en-as ka-en en-ka be-en en-be eo-en en-eo cy-en en-cy ga-en en-ga ug-en en-ug az-en en-az xh-en en-xh af-en en-af oc-en en-oc br-en en-br rw-en en-rw km-en en-km ku-en en-ku wa-en en-wa mg-en en-mg kk-en en-kk tg-en en-tg am-en en-am ps-en en-ps my-en en-my uz-en en-uz ur-en en-ur ky-en en-ky gd-en en-gd sh-en en-sh li-en en-li zu-en en-zu fy-en en-fy tk-en en-tk yi-en en-yi tt-en en-tt se-en en-se ha-en en-ha ig-en en-ig)
pairs_num=187

type=${1:-"baseline"}
step=${2:-3000}
if [ ${type} == "baseline" ];then
    exp_name=opus_baseline
    model_name=checkpoint_best
elif [ ${type} == "unions" ];then
    exp_name=opus_baseline/continue_train.continue_step10000.negative_loss_lambda1.0.translation_loss_lambda1.0.lr0.00005.wu1
    model_name=checkpoint_1_${step}
fi

save_dir=$opus_checkpoints/$exp_name
for cur_id in $(seq 0 ${pairs_num}); do
    lang_pair=${lang_pair_list[$cur_id]}
    gpu_id=$(( $cur_id % 8 ))

    SRC=${lang_pair: 0: 2}
    TGT=${lang_pair: 3: 2}
    MODEL_PATH=$save_dir/${model_name}.pt

    if [ $SRC == en ];then
        FTGT=${DATA_DIR}/raw/test.en-${TGT}.${TGT}
    else
        FTGT=${DATA_DIR}/raw/test.en-${SRC}.en
    fi

    mkdir -p $save_dir/gen_log

    CUDA_VISIBLE_DEVICES=$gpu_id fairseq-generate ${DATA_DIR}/data-bin \
        --task translation_multi_simple_epoch --encoder-langtok src --decoder-langtok --path $MODEL_PATH \
        --langs af,am,ar,as,az,be,bg,bn,br,bs,ca,cs,cy,da,de,el,en,eo,es,et,eu,fa,fi,fr,fy,ga,gd,gl,gu,ha,he,hi,hr,hu,id,ig,is,it,ja,ka,kk,km,kn,ko,ku,ky,li,lt,lv,mg,mk,ml,mr,ms,mt,my,nb,ne,nl,nn,no,oc,or,pa,pl,ps,pt,ro,ru,rw,se,sh,si,sk,sl,sq,sr,sv,ta,te,tg,th,tk,tr,tt,ug,uk,ur,uz,vi,wa,xh,yi,zh,zu \
        --lang-pairs $lang_pairs \
        --source-lang $SRC --target-lang $TGT --max-tokens 32768 --gen-subset test \
        --beam 5 --lenpen 1.0 --remove-bpe=sentencepiece > $save_dir/gen_log/${SRC}-${TGT}.gen_log 2>&1 & 

    if [ $gpu_id == 7 ];then
        wait 
    fi
done

rm $opus_checkpoints/$exp_name/en_xx_result.csv
rm $opus_checkpoints/$exp_name/xx_en_result.csv
for cur_id in $(seq 0 ${pairs_num}); do
    lang_pair=${lang_pair_list[$cur_id]}
    SRC=${lang_pair: 0: 2}
    TGT=${lang_pair: 3: 2}

    if [ $SRC == en ];then
        FTGT=${DATA_DIR}/raw/test.en-${TGT}.${TGT}
        Fresult=$opus_checkpoints/$exp_name/en_xx_result.csv
    elif [ $TGT == en ];then
        FTGT=${DATA_DIR}/raw/test.en-${SRC}.en
        Fresult=$opus_checkpoints/$exp_name/xx_en_result.csv
    fi
    echo ">> ${SRC}_${TGT} processing" 
    cat $opus_checkpoints/$exp_name/gen_log/${SRC}-${TGT}.gen_log | grep -P "^H" |sort -V | cut -f 3- > $opus_checkpoints/$exp_name/gen_log/${SRC}-${TGT}.hyp
    printf "${SRC}_${TGT}, " >> $Fresult
    sacrebleu $FTGT -i $opus_checkpoints/$exp_name/gen_log/${SRC}-${TGT}.hyp --language-pair ${SRC}-${TGT} -b -w 2 >> $Fresult
    echo ">> ${SRC}_${TGT} finised"
done
