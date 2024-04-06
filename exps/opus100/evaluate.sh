
# evaluate baseline
bash compute_sacrebleu_ours_supervised.sh baseline
gpu_id=0
bash compute_sacrebleu_zst.sh baseline $gpu_id

# select model
dev_data_path=
model_path=
data_path=
CUDA_VISIBLE_DEVICES=2 python model_selector.py $dev_data_path $model_path $data_path

# evaluate unions
gpu_id=0
step=
bash compute_sacrebleu_ours.sh unions $gpu_id ${step}
wait 
bash compute_sacrebleu_ours_supervised.sh unions ${step}

