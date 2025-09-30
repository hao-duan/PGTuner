#!/usr/bin/env bash
set -euo pipefail

# ===== paremeters:（default: tiny, main）=====
DATASET_NAME="${1:-tiny}"
EXPERIMENT_MODE="${2:-main}"

ROOT="$(cd -- "$(dirname "$0")" && pwd -P)"

log() { echo -e "\033[1;32m[$(date '+%F %T')] $*\033[0m"; }
run() { log "$*"; "$@"; }
trap 'echo -e "\033[1;31m[ABORT] Command failed on line $LINENO\033[0m"' ERR

pushd "$ROOT/query_performance_predict" >/dev/null
run python active_learning.py --dataset-name "$DATASET_NAME" --experiment-mode "$EXPERIMENT_MODE"
popd >/dev/null

pushd "$ROOT/parameter_configuration_recommend" >/dev/null

MODEL_DIR="./[128, 256, 256, 64]_[256, 256, 256, 64]_TD3/model_params"

run cp -n "$MODEL_DIR/pre-trained_actor_8000_5000_128_1e-05_0.0001_0.2_2_1_200.pth" \
          "$MODEL_DIR/actor_8000_5000_128_1e-05_0.0001_0.2_2_1_200.pth"
run cp -n "$MODEL_DIR/pre-trained_critic1_8000_5000_128_1e-05_0.0001_0.2_2_1_200.pth" \
          "$MODEL_DIR/critic1_8000_5000_128_1e-05_0.0001_0.2_2_1_200.pth"
run cp -n "$MODEL_DIR/pre-trained_critic2_8000_5000_128_1e-05_0.0001_0.2_2_1_200.pth" \
          "$MODEL_DIR/critic2_8000_5000_128_1e-05_0.0001_0.2_2_1_200.pth"

run python evaluate.py --dataset-name "$DATASET_NAME" --experiment-mode "$EXPERIMENT_MODE"
run python generate_recommended_configurations.py --dataset-name "$DATASET_NAME" --experiment-mode "$EXPERIMENT_MODE"
popd >/dev/null

run python "$ROOT/query_performance_verify.py" --dataset-name "$DATASET_NAME" --experiment-mode "$EXPERIMENT_MODE"

log "Pipeline finished successfully for dataset=${DATASET_NAME}, experiment mode=${EXPERIMENT_MODE}."
