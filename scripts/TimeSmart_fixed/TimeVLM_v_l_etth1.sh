export TOKENIZERS_PARALLELISM=false
model_name=TimeVLM_v_l
vlm_type=clip
gpu=0
image_size=56
norm_const=0.4
three_channel_image=True
finetune_vlm=False
batch_size=32
num_workers=32
learning_rate=0.001
seq_len=512
percent=1
train_epochs=30
results_path=TimeVLM_fix_l.txt
results_folder=./Result_TimeVLM_fix/
checkpoints=./Checkpoint_TimeVLM_fix/

# Create Logs_TimeVLM_fix directory if it doesn't exist
if [ ! -d "Logs_TimeVLM_fix" ]; then
    mkdir Logs_TimeVLM_fix
fi

# Supports both few-shot (percent < 1.0) and full-shot (percent = 1.0)
run_experiment() {
    local dset=$1
    local data=$2
    local n_vars=$3
    local pred_len=$4
    local periodicity=$5
    local dropout=$6


    # Determine task name based on percent
    local task_name="few_shot_forecast"
    if [ "$percent" = "1" ]; then
        task_name="long_term_forecast"
    fi

    log_file="Logs_TimeVLM_fix/${model_name}_${dset}_${seq_len}_${pred_len}_${percent}p_${dropout}.log"
    echo "Running experiment: dataset=${dset}, seq_len=${seq_len}, pred_len=${pred_len}, percent=${percent}, dropout=${dropout}"

    # Use runNew.py instead of runNew.py
    python -u runNew.py \
      --task_name $task_name \
      --is_training 1 \
      --root_path ./dataset/ \
      --data_path ${dset}.csv \
      --model_id ${dset}_${seq_len}_${pred_len} \
      --model $model_name \
      --data ${data} \
      --dset ${dset} \
      --features M \
      --seq_len $seq_len \
      --label_len 48 \
      --pred_len $pred_len \
      --enc_in $n_vars \
      --dec_in $n_vars \
      --c_out $n_vars \
      --des 'Exp' \
      --itr 1 \
      --gpu $gpu \
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
      --checkpoints $checkpoints \
      --percent $percent > $log_file
}

# ETTh1, n_vars=7, periodicity=24
run_experiment ETTh1_opt ETTh1 4 96 24 0.1
run_experiment ETTh1_opt ETTh1 4 192 24 0.1
run_experiment ETTh1_opt ETTh1 4 336 24 0.1
run_experiment ETTh1_opt ETTh1 4 720 24 0.1