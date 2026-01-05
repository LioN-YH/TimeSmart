#!/bin/sh

export TOKENIZERS_PARALLELISM=false

# Fixed hyperparameters
model_name=VLMtest
gpu=2
vlm_type=vilt
image_size=56
norm_const=0.4
three_channel_image=True
finetune_vlm=False
batch_size=32
num_workers=32
learning_rate=0.001
seq_len=512
percent=1
train_epochs=15
results_path=VLMtest.txt
results_folder=./Result_VLMtest/



# Create logs directory if it doesn't exist
if [ ! -d "logs_test" ]; then
    mkdir logs_test
fi
# Run a single experiment (only varying dropout)
run_experiment() {
    dset=$1
    data=$2
    n_vars=$3
    pred_len=$4
    periodicity=$5
    dropout=$6


    echo "Starting experiment on GPU $gpu_id: ${dset}_${pred_len}_do${dropout}"

    task_name="long_term_forecast"

    log_file="logs_test/${model_name}_${vlm_type}_${dset}_${seq_len}_${pred_len}_do${dropout}_${percent}p.log"

    python -u run.py \
      --task_name $task_name \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path ${dset}.csv \
      --model_id ${dset}_${seq_len}_${pred_len}_do${dropout} \
      --model $model_name \
      --data $data \
      --features M \
      --seq_len $seq_len \
      --label_len 48 \
      --pred_len $pred_len \
      --enc_in $n_vars \
      --dec_in $n_vars \
      --c_out $n_vars \
      --gpu $gpu \
      --des 'Dropout Hyperparameter Search' \
      --itr 1 \
      --use_amp \
      --train_epochs $train_epochs \
      --image_size $image_size \
      --norm_const $norm_const \
      --periodicity $periodicity \
      --three_channel_image $three_channel_image \
      --finetune_vlm $finetune_vlm \
      --batch_size $batch_size \
      --learning_rate $learning_rate \
      --num_workers $num_workers \
      --vlm_type $vlm_type \
      --dropout $dropout \
      --results_path $results_path \
      --results_folder $results_folder \
      --percent $percent > "$log_file" 2>&1

    echo "Experiment completed on GPU $gpu_id: ${dset}_${pred_len}_do${dropout}"
}

# ETTh1, n_vars=7, periodicity=24
run_experiment ETTh1 ETTh1 7 96 24 0.1
# run_experiment Electricity custom 321 96 24 0.1