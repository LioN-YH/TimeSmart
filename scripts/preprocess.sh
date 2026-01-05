num_workers=32
batch_size=1
norm_const=0.4
image_size=56
seq_len=512
save_folder=./ImageTensors/

# Create Logs_pre directory if it doesn't exist
if [ ! -d "Logs_pre" ]; then
    mkdir Logs_pre
fi

# Supports both few-shot (percent < 1.0) and full-shot (percent = 1.0)
run_preprocess() {
    local dataset=$1
    local data=$2
    local periodicity=$3
    local T_x=$4
    local pred_len=$5
    
    log_file="Logs_pre/${dataset}_${pred_len}.log"
    echo "Running preprocess: dataset=${dataset}, pred_len=${pred_len}"

    python -u src/TimeSmart/preprocess.py \
      --num_workers $num_workers \
      --batch_size $batch_size \
      --dataset $dataset \
      --data $data \
      --root_path ./dataset/ \
      --data_path ${dataset}.csv \
      --features M \
      --seq_len $seq_len \
      --label_len 48 \
      --pred_len $pred_len \
      --image_size $image_size \
      --norm_const $norm_const \
      --periodicity $periodicity \
      --T_x $T_x \
      --save_folder $save_folder > $log_file
}

# ETTh1
run_preprocess ETTh1 ETTh1 24 3600 96
run_preprocess ETTh1 ETTh1 24 3600 192
run_preprocess ETTh1 ETTh1 24 3600 336
run_preprocess ETTh1 ETTh1 24 3600 720

# ETTh2
run_preprocess ETTh2 ETTh2 24 3600 96
run_preprocess ETTh2 ETTh2 24 3600 192
run_preprocess ETTh2 ETTh2 24 3600 336
run_preprocess ETTh2 ETTh2 24 3600 720

# ETTm1
run_preprocess ETTm1 ETTm1 96 900 96
run_preprocess ETTm1 ETTm1 96 900 192
run_preprocess ETTm1 ETTm1 96 900 336
run_preprocess ETTm1 ETTm1 96 900 720

# ETTm2
run_preprocess ETTm2 ETTm2 96 900 96
run_preprocess ETTm2 ETTm2 96 900 192
run_preprocess ETTm2 ETTm2 96 900 336
run_preprocess ETTm2 ETTm2 96 900 720

# Electricity
run_preprocess Electricity custom 24 3600 96
run_preprocess Electricity custom 24 3600 192
run_preprocess Electricity custom 24 3600 336
run_preprocess Electricity custom 24 3600 720

# Traffic
run_preprocess Traffic custom 24 3600 96
run_preprocess Traffic custom 24 3600 192
run_preprocess Traffic custom 24 3600 336
run_preprocess Traffic custom 24 3600 720

# Weather
run_preprocess Weather custom 144 600 96
run_preprocess Weather custom 144 600 192
run_preprocess Weather custom 144 600 336
run_preprocess Weather custom 144 600 720