#!/bin/sh

export TOKENIZERS_PARALLELISM=false

# Fixed hyperparameters
model_name=TimeVLM_v_l
vlm_type=vilt
image_size=56
norm_const=0.4
three_channel_image=True
finetune_vlm=False
batch_size=8
num_workers=32
learning_rate=0.001
seq_len=512
percent=1
train_epochs=15
results_path=TimeVLM_v_l_ELC.txt
results_folder=./Result_TimeVLM_v_l/

# GPU configuration - you have 2 GPUs (0,1)
available_gpus="2"
max_parallel_jobs=1

# Create logs directory if it doesn't exist
if [ ! -d "logs_v_l" ]; then
    mkdir logs_v_l
fi

if [ ! -d "logs_v_l/hyperparameter_search" ]; then
    mkdir -p logs_v_l/hyperparameter_search
fi

# Function to get next available GPU (round-robin)
get_next_gpu() {
    gpus="$1"
    job_index=$2
    num_gpus=$(echo $gpus | wc -w)
    echo $gpus | cut -d' ' -f$(( (job_index % num_gpus) + 1 ))
}

# Run a single experiment (only varying dropout)
run_single_experiment() {
    dset=$1
    data=$2
    n_vars=$3
    pred_len=$4
    periodicity=$5
    dropout=$6
    gpu_id=$7

    echo "Starting experiment on GPU $gpu_id: ${dset}_${pred_len}_do${dropout}"

    task_name="long_term_forecast"

    log_file="logs_v_l/hyperparameter_search/${model_name}_${dset}_${seq_len}_${pred_len}_do${dropout}_${percent}p.log"

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
      --des 'Dropout Hyperparameter Search' \
      --itr 1 \
      --gpu $gpu_id \
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

# Launch all experiments for one (dset, pred_len) with different dropouts
launch_experiments_for_config() {
    dset=$1
    data=$2
    n_vars=$3
    pred_len=$4
    periodicity=$5

    echo "Launching experiments for ${dset} with pred_len=${pred_len} (dropout only)"
    dropouts="0.1 0.2 0.3 0.4 0.5"

    job_count=0
    running_pids=()

    for dropout in $dropouts; do
        gpu_id=$(get_next_gpu "$available_gpus" $job_count)
        run_single_experiment $dset $data $n_vars $pred_len $periodicity $dropout $gpu_id &
        pid=$!
        running_pids+=($pid)
        echo "Started dropout=${dropout} on GPU $gpu_id (PID: $pid)"
        job_count=$((job_count + 1))

        # If we hit max_parallel_jobs, wait for one batch to finish
        if [ ${#running_pids[@]} -ge $max_parallel_jobs ]; then
            echo "Waiting for current batch (${#running_pids[@]} jobs) to complete..."
            for p in "${running_pids[@]}"; do
                wait $p
            done
            running_pids=()
            echo "Batch completed."
        fi
    done

    # Wait for remaining jobs
    if [ ${#running_pids[@]} -gt 0 ]; then
        echo "Waiting for remaining ${#running_pids[@]} jobs to complete..."
        for p in "${running_pids[@]}"; do
            wait $p
        done
        echo "All jobs for ${dset}_pl${pred_len} completed."
    fi
}

# Dataset configurations
datasets="Electricity"
data_types="custom"
n_vars_list="321"
periodicities="24"
pred_lengths="96 192 336 720"

# Main execution
echo "Starting DROPOUT-ONLY hyperparameter search (skipping ETTh1 + pred_len=96)"
echo "Total datasets: 1, pred lengths: 4, dropouts: 5 → up to 20 experiments"
echo "Max parallel jobs: $max_parallel_jobs on GPUs: $available_gpus"
echo ""

dataset_index=1
for dset in $datasets; do
    data_type=$(echo $data_types | awk -v i=$dataset_index '{print $i}')
    n_vars=$(echo $n_vars_list | awk -v i=$dataset_index '{print $i}')
    periodicity=$(echo $periodicities | awk -v i=$dataset_index '{print $i}')

    echo "Processing dataset: ${dset} (n_vars=${n_vars}, periodicity=${periodicity})"
    echo "------------------------------------------------------------"

    for pred_len in $pred_lengths; do
        # ✅ Skip already completed experiment: ETTh1 with pred_len=96
        if [ "$dset" = "ETTh1" ] && [ "$pred_len" -eq 96 ]; then
            echo "Skipping ${dset} with pred_len=${pred_len} (already completed)"
            continue
        fi

        launch_experiments_for_config $dset $data_type $n_vars $pred_len $periodicity
    done

    dataset_index=$((dataset_index + 1))
done

echo "ALL EXPERIMENTS LAUNCHED AND COMPLETED!"
echo "Logs saved under: logs_v_l/hyperparameter_search/"