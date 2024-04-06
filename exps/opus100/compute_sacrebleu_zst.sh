export LC_ALL=C.UTF-8
export NCCL_IB_DISABLE=1
set -x

spm_decode=PATH_TO_SPM_DECODE
spm_model=PATH_TO_SPM_MODEL/spm_64k.model
OPUS_HOME=PATH_TO_SAVE_OUTPUT/opus_100
opus_checkpoints=$OPUS_HOME/checkpoints
DATA_DIR=DATA_HOME/opus_100

lang_pair_list=(es-en en-es fr-en en-fr ro-en en-ro nl-en en-nl cs-en en-cs el-en en-el hu-en en-hu pl-en en-pl tr-en en-tr pt-en en-pt bg-en en-bg it-en en-it fi-en en-fi hr-en en-hr ar-en en-ar sr-en en-sr he-en en-he de-en en-de sl-en en-sl ru-en en-ru sv-en en-sv da-en en-da et-en en-et bs-en en-bs sk-en en-sk id-en en-id no-en en-no fa-en en-fa lt-en en-lt zh-en en-zh lv-en en-lv mk-en en-mk vi-en en-vi th-en en-th ja-en en-ja sq-en en-sq ms-en en-ms is-en en-is ko-en en-ko uk-en en-uk ca-en en-ca eu-en en-eu mt-en en-mt gl-en en-gl ml-en en-ml bn-en en-bn pa-en en-pa hi-en en-hi ta-en en-ta si-en en-si nb-en en-nb nn-en en-nn te-en en-te gu-en en-gu mr-en en-mr ne-en en-ne kn-en en-kn or-en en-or as-en en-as ka-en en-ka be-en en-be eo-en en-eo cy-en en-cy ga-en en-ga ug-en en-ug az-en en-az xh-en en-xh af-en en-af oc-en en-oc br-en en-br rw-en en-rw km-en en-km ku-en en-ku wa-en en-wa mg-en en-mg kk-en en-kk tg-en en-tg am-en en-am ps-en en-ps my-en en-my uz-en en-uz ur-en en-ur ky-en en-ky gd-en en-gd sh-en en-sh li-en en-li zu-en en-zu fy-en en-fy tk-en en-tk yi-en en-yi tt-en en-tt se-en en-se ha-en en-ha ig-en en-ig)
pairs_num=187

type=${1:-"baseline"}
gpu_id=${2:-0}
step=${3:-1000}
if [ ${type} == "baseline" ];then
    exp_name=opus_baseline
    model_name=checkpoint_best
elif [ ${type} == "unions" ];then
    exp_name=opus_baseline/continue_train.continue_step10000.negative_loss_lambda1.0.translation_loss_lambda1.0.lr0.00005.wu1
    model_name=checkpoint_1_${step}
fi

save_dir=$opus_checkpoints/$exp_name
DAT_BIN=$DATA_DIR/data-bin
spm_dir=$DATA_DIR/spm_data
MODEL_DIR=$save_dir
MODEL_PATH=$save_dir/${model_name}.pt
FOUT_dir=$save_dir/${model_name}_zero-shot
mkdir -p $FOUT_dir
Fresult=$opus_checkpoints/$exp_name/${model_name}_zero_shot_result.csv
F_otr_result=$opus_checkpoints/$exp_name/${model_name}_zero_shot_otr_result.csv

echo "${model_name}:" > $Fresult
echo "${model_name}:" > $F_otr_result

lpairs=(de-fr fr-ru de-nl ru-zh ar-zh ar-nl)

for lpair in ${lpairs[@]} ;do
    TMP=(${lpair//-/ })
    SRC=${TMP[0]}
    TGT=${TMP[1]}
    cp $DAT_BIN/dict.$SRC.txt $MODEL_DIR
    cp $DAT_BIN/dict.$TGT.txt $MODEL_DIR
    FSRC=${spm_dir}/test.${SRC}-${TGT}.${SRC}
    FTGT=${DATA_DIR}/raw/test.${SRC}-${TGT}.${TGT}
    FOUT=${FOUT_dir}/test.${SRC}-${TGT}.hyp.${TGT}

    # $SRC-$TGT evaluation
    cat $FSRC | \
    CUDA_VISIBLE_DEVICES=${gpu_id} fairseq-interactive ${MODEL_DIR} \
        --task translation_multi_simple_epoch --encoder-langtok src --decoder-langtok --path $MODEL_PATH \
        --langs af,am,ar,as,az,be,bg,bn,br,bs,ca,cs,cy,da,de,el,en,eo,es,et,eu,fa,fi,fr,fy,ga,gd,gl,gu,ha,he,hi,hr,hu,id,ig,is,it,ja,ka,kk,km,kn,ko,ku,ky,li,lt,lv,mg,mk,ml,mr,ms,mt,my,nb,ne,nl,nn,no,oc,or,pa,pl,ps,pt,ro,ru,rw,se,sh,si,sk,sl,sq,sr,sv,ta,te,tg,th,tk,tr,tt,ug,uk,ur,uz,vi,wa,xh,yi,zh,zu \
        --lang-pairs es-en,en-es,fr-en,en-fr,ro-en,en-ro,nl-en,en-nl,cs-en,en-cs,el-en,en-el,hu-en,en-hu,pl-en,en-pl,tr-en,en-tr,pt-en,en-pt,bg-en,en-bg,it-en,en-it,fi-en,en-fi,hr-en,en-hr,ar-en,en-ar,sr-en,en-sr,he-en,en-he,de-en,en-de,sl-en,en-sl,ru-en,en-ru,sv-en,en-sv,da-en,en-da,et-en,en-et,bs-en,en-bs,sk-en,en-sk,id-en,en-id,no-en,en-no,fa-en,en-fa,lt-en,en-lt,zh-en,en-zh,lv-en,en-lv,mk-en,en-mk,vi-en,en-vi,th-en,en-th,ja-en,en-ja,sq-en,en-sq,ms-en,en-ms,is-en,en-is,ko-en,en-ko,uk-en,en-uk,ca-en,en-ca,eu-en,en-eu,mt-en,en-mt,gl-en,en-gl,ml-en,en-ml,bn-en,en-bn,pa-en,en-pa,hi-en,en-hi,ta-en,en-ta,si-en,en-si,nb-en,en-nb,nn-en,en-nn,te-en,en-te,gu-en,en-gu,mr-en,en-mr,ne-en,en-ne,kn-en,en-kn,or-en,en-or,as-en,en-as,ka-en,en-ka,be-en,en-be,eo-en,en-eo,cy-en,en-cy,ga-en,en-ga,ug-en,en-ug,az-en,en-az,xh-en,en-xh,af-en,en-af,oc-en,en-oc,br-en,en-br,rw-en,en-rw,km-en,en-km,ku-en,en-ku,wa-en,en-wa,mg-en,en-mg,kk-en,en-kk,tg-en,en-tg,am-en,en-am,ps-en,en-ps,my-en,en-my,uz-en,en-uz,ur-en,en-ur,ky-en,en-ky,gd-en,en-gd,sh-en,en-sh,li-en,en-li,zu-en,en-zu,fy-en,en-fy,tk-en,en-tk,yi-en,en-yi,tt-en,en-tt,se-en,en-se,ha-en,en-ha,ig-en,en-ig \
        --source-lang $SRC --target-lang $TGT --buffer-size 1024 --batch-size 128 \
        --beam 5 --lenpen 1.0 --remove-bpe=sentencepiece --no-progress-bar > ${FOUT_dir}/${SRC}-${TGT}.gen_log
    cat ${FOUT_dir}/${SRC}-${TGT}.gen_log | grep -P "^H" | cut -f 3- > $FOUT

    printf "${SRC}_${TGT}, " >> $Fresult
    sacrebleu $FTGT -i $FOUT -l $lpair -b -w 2 >> $Fresult
    printf "${SRC}_${TGT}, " >> $F_otr_result
    python lid.py --src_path ${FOUT} --lang ${TGT} >> $F_otr_result


    # $TGT-$SRC evaluation
    FSRC=${spm_dir}/test.${SRC}-${TGT}.${TGT}
    FTGT=${DATA_DIR}/raw/test.${SRC}-${TGT}.${SRC}
    FOUT=${FOUT_dir}/test.${TGT}-${SRC}.hyp.${SRC}
    cat $FSRC | \
    CUDA_VISIBLE_DEVICES=${gpu_id} fairseq-interactive ${MODEL_DIR} \
        --task translation_multi_simple_epoch --encoder-langtok src --decoder-langtok --path $MODEL_PATH \
        --langs af,am,ar,as,az,be,bg,bn,br,bs,ca,cs,cy,da,de,el,en,eo,es,et,eu,fa,fi,fr,fy,ga,gd,gl,gu,ha,he,hi,hr,hu,id,ig,is,it,ja,ka,kk,km,kn,ko,ku,ky,li,lt,lv,mg,mk,ml,mr,ms,mt,my,nb,ne,nl,nn,no,oc,or,pa,pl,ps,pt,ro,ru,rw,se,sh,si,sk,sl,sq,sr,sv,ta,te,tg,th,tk,tr,tt,ug,uk,ur,uz,vi,wa,xh,yi,zh,zu \
        --lang-pairs es-en,en-es,fr-en,en-fr,ro-en,en-ro,nl-en,en-nl,cs-en,en-cs,el-en,en-el,hu-en,en-hu,pl-en,en-pl,tr-en,en-tr,pt-en,en-pt,bg-en,en-bg,it-en,en-it,fi-en,en-fi,hr-en,en-hr,ar-en,en-ar,sr-en,en-sr,he-en,en-he,de-en,en-de,sl-en,en-sl,ru-en,en-ru,sv-en,en-sv,da-en,en-da,et-en,en-et,bs-en,en-bs,sk-en,en-sk,id-en,en-id,no-en,en-no,fa-en,en-fa,lt-en,en-lt,zh-en,en-zh,lv-en,en-lv,mk-en,en-mk,vi-en,en-vi,th-en,en-th,ja-en,en-ja,sq-en,en-sq,ms-en,en-ms,is-en,en-is,ko-en,en-ko,uk-en,en-uk,ca-en,en-ca,eu-en,en-eu,mt-en,en-mt,gl-en,en-gl,ml-en,en-ml,bn-en,en-bn,pa-en,en-pa,hi-en,en-hi,ta-en,en-ta,si-en,en-si,nb-en,en-nb,nn-en,en-nn,te-en,en-te,gu-en,en-gu,mr-en,en-mr,ne-en,en-ne,kn-en,en-kn,or-en,en-or,as-en,en-as,ka-en,en-ka,be-en,en-be,eo-en,en-eo,cy-en,en-cy,ga-en,en-ga,ug-en,en-ug,az-en,en-az,xh-en,en-xh,af-en,en-af,oc-en,en-oc,br-en,en-br,rw-en,en-rw,km-en,en-km,ku-en,en-ku,wa-en,en-wa,mg-en,en-mg,kk-en,en-kk,tg-en,en-tg,am-en,en-am,ps-en,en-ps,my-en,en-my,uz-en,en-uz,ur-en,en-ur,ky-en,en-ky,gd-en,en-gd,sh-en,en-sh,li-en,en-li,zu-en,en-zu,fy-en,en-fy,tk-en,en-tk,yi-en,en-yi,tt-en,en-tt,se-en,en-se,ha-en,en-ha,ig-en,en-ig \
        --source-lang $TGT --target-lang $SRC --buffer-size 1024 --batch-size 128 \
        --beam 5 --lenpen 1.0 --remove-bpe=sentencepiece --no-progress-bar > ${FOUT_dir}/${TGT}-${SRC}.gen_log
    cat ${FOUT_dir}/${TGT}-${SRC}.gen_log | grep -P "^H" | cut -f 3- > $FOUT
    printf "${TGT}_${SRC}, " >> $Fresult
    sacrebleu $FTGT -i $FOUT -l ${TGT}-${SRC} -b -w 2 >> $Fresult
    printf "${TGT}_${SRC}, " >> $F_otr_result
    python lid.py --src_path ${FOUT} --lang ${SRC} >> $F_otr_result
done

