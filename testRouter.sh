#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-./router_stats}"
mkdir -p "$OUT_DIR"

COMMON_ARGS=(
  --features M
  --target OT
  --seq_len 512
  --label_len 48
  --pred_len 96
  --periodicity 24
  --router_temperature 5.0
  --router_bias 0.0
  --batch_size 256
  --num_workers 0
  --router_bias 0.20
  --trend_low_freq_weight 0.65
  --trend_direction_weight 0.15
  --trend_linearity_weight 0.20
)

python src/TimeApart/computeRouter.py \
  --csv_path dataset/ETTh1_seasonal.csv \
  --dataset_name ETTh1_seasonal \
  --output_json "$OUT_DIR/ETTh1_seasonal_router_stats_pred96.json" \
  "${COMMON_ARGS[@]}"

python src/TimeApart/computeRouter.py \
  --csv_path dataset/OT_trend.csv \
  --dataset_name OT_trend \
  --output_json "$OUT_DIR/OT_trend_router_stats_pred96.json" \
  "${COMMON_ARGS[@]}"

echo "[OK] outputs written to $OUT_DIR"