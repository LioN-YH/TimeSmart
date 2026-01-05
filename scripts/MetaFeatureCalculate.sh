#!/usr/bin/env bash
export TOKENIZERS_PARALLELISM=false

# 基础配置
gpu=0
output_dir=./dataset/Meta
mkdir -p "${output_dir}" Logs_mfe

# 单次运行：计算并保存 {dset}-{seq_len}-{pred_len}.json 到 output_dir
run_mfe() {
  local dset=$1        # 例如：ETTh1
  local data=$2        # 例如：ETTh1（用于 data_provider 选择）
  local seq_len=$3     # 回望窗口长度
  local pred_len=$4    # 预测长度

  local log_file="Logs_mfe/mfe_${dset}_${seq_len}_${pred_len}.log"
  echo "Computing meta-features: dset=${dset}, data=${data}, seq_len=${seq_len}, pred_len=${pred_len}"

  CUDA_VISIBLE_DEVICES=${gpu} \
  python -u precompute_meta_features.py \
    --data "${data}" \
    --dset "${dset}" \
    --seq_len "${seq_len}" \
    --pred_len "${pred_len}" \
    --output "${output_dir}" > "${log_file}" 2>&1
}

run_mfe ETTh1 ETTh1 512 96
run_mfe ETTh1 ETTh1 512 192
run_mfe ETTh1 ETTh1 512 336
run_mfe ETTh1 ETTh1 512 720

run_mfe ETTh2 ETTh2 512 96
run_mfe ETTh2 ETTh2 512 192
run_mfe ETTh2 ETTh2 512 336
run_mfe ETTh2 ETTh2 512 720

run_mfe ETTm1 ETTm1 512 96
run_mfe ETTm1 ETTm1 512 192
run_mfe ETTm1 ETTm1 512 336
run_mfe ETTm1 ETTm1 512 720

run_mfe ETTm2 ETTm2 512 96
run_mfe ETTm2 ETTm2 512 192
run_mfe ETTm2 ETTm2 512 336
run_mfe ETTm2 ETTm2 512 720

# run_mfe Electricity custom 512 96
# run_mfe Electricity custom 512 192
# run_mfe Electricity custom 512 336
# run_mfe Electricity custom 512 720

# run_mfe Weather custom 512 96
# run_mfe Weather custom 512 192
# run_mfe Weather custom 512 336
# run_mfe Weather custom 512 720

# run_mfe Traffic custom 512 96
# run_mfe Traffic custom 512 192
# run_mfe Traffic custom 512 336
# run_mfe Traffic custom 512 720

# # 扩展：多 seq_len × 多 pred_len 批量运行
# seq_len_list=(256 512 1024)
# pred_len_list=(96 192 336 720)
# datasets=(ETTh1 ETTh2 ETTm1 ETTm2)

# for d in "${datasets[@]}"; do
#   for s in "${seq_len_list[@]}"; do
#     for p in "${pred_len_list[@]}"; do
#       run_mfe "${d}" "${d}" "${s}" "${p}"
#     done
#   done
# done