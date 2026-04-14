#!/usr/bin/env bash
set -euo pipefail

# =====================
# Phase 1: Model Inference
# =====================
PYTHON_BIN="python"
PROJECT_ROOT="/egr/research-optml/chenyiw9/projects/VLGuard"
CKPT_ROOT="/egr/research-optml/chenyiw9/projects/LLaVA/checkpoints_new"

# Number of questions to sample per dataset (0 = use all, which is the default).
# Can be overridden on the command line: ./run_eval_each_model.sh [MAX_QUESTIONS] [SAMPLE_SEED]
# Examples:
#   ./run_eval_each_model.sh          # use all questions
#   ./run_eval_each_model.sh 128      # sample 128 per dataset
#   ./run_eval_each_model.sh 128 0    # sample 128, seed=0
MAX_QUESTIONS="${1:-0}"
SAMPLE_SEED="${2:-42}"

# All 8 available GPUs
GPU_IDS=(0 1 2 3 4 5 6 7)

# =====================
# Models to evaluate
# Format: "engine_name|model_path"
# =====================
MODELS=(
  # --- NPO Models (checkpoints-llava-retain-mix-npo-search-full-2) ---
  # beta=0.8
  "llava-7b_npo_beta-0.8_retainA-1.0_lr-4.5e-7|$CKPT_ROOT/checkpoints-llava-retain-mix-npo-search-full-2/llava-7b_npo_beta-0.8_retainA-1.0_lr-4.5e-7"
  "llava-7b_npo_beta-0.8_retainA-1.0_lr-4e-7|$CKPT_ROOT/checkpoints-llava-retain-mix-npo-search-full-2/llava-7b_npo_beta-0.8_retainA-1.0_lr-4e-7"
  # beta=0.85
  "llava-7b_npo_beta-0.85_retainA-1.0_lr-4.5e-7|$CKPT_ROOT/checkpoints-llava-retain-mix-npo-search-full-2/llava-7b_npo_beta-0.85_retainA-1.0_lr-4.5e-7"
  "llava-7b_npo_beta-0.85_retainA-1.0_lr-4e-7|$CKPT_ROOT/checkpoints-llava-retain-mix-npo-search-full-2/llava-7b_npo_beta-0.85_retainA-1.0_lr-4e-7"
  "llava-7b_npo_beta-0.85_retainA-1.0_lr-5e-7|$CKPT_ROOT/checkpoints-llava-retain-mix-npo-search-full-2/llava-7b_npo_beta-0.85_retainA-1.0_lr-5e-7"
  # beta=0.9
  "llava-7b_npo_beta-0.9_retainA-1.0_lr-4.5e-7|$CKPT_ROOT/checkpoints-llava-retain-mix-npo-search-full-2/llava-7b_npo_beta-0.9_retainA-1.0_lr-4.5e-7"
  "llava-7b_npo_beta-0.9_retainA-1.0_lr-4e-7|$CKPT_ROOT/checkpoints-llava-retain-mix-npo-search-full-2/llava-7b_npo_beta-0.9_retainA-1.0_lr-4e-7"
  "llava-7b_npo_beta-0.9_retainA-1.0_lr-5e-7|$CKPT_ROOT/checkpoints-llava-retain-mix-npo-search-full-2/llava-7b_npo_beta-0.9_retainA-1.0_lr-5e-7"
  # beta=0.95
  "llava-7b_npo_beta-0.95_retainA-1.0_lr-4.5e-7|$CKPT_ROOT/checkpoints-llava-retain-mix-npo-search-full-2/llava-7b_npo_beta-0.95_retainA-1.0_lr-4.5e-7"
  "llava-7b_npo_beta-0.95_retainA-1.0_lr-4e-7|$CKPT_ROOT/checkpoints-llava-retain-mix-npo-search-full-2/llava-7b_npo_beta-0.95_retainA-1.0_lr-4e-7"
  "llava-7b_npo_beta-0.95_retainA-1.0_lr-5e-7|$CKPT_ROOT/checkpoints-llava-retain-mix-npo-search-full-2/llava-7b_npo_beta-0.95_retainA-1.0_lr-5e-7"
)

# =====================
# Common paths
# =====================
EVAL_PY="$PROJECT_ROOT/VLGuard_eval.py"
IMAGE_DIR="/egr/research-optml/chenyiw9/datasets/VLGuard_dataset/test"

META_1="$PROJECT_ROOT/safety_data/test_share_1shot.json"   # jailbreak safe_safes
DATASET_1="safe_safes"
OUTDIR_1="$PROJECT_ROOT/results/jailbreak"

META_2="$PROJECT_ROOT/safety_data/test_what_3shot.json"    # jailbreak unsafes
DATASET_2="unsafes"
OUTDIR_2="$PROJECT_ROOT/results/jailbreak"

META_3="$PROJECT_ROOT/data/test.json"                       # normal safe_safes
DATASET_3="safe_safes"
OUTDIR_3="$PROJECT_ROOT/results/normal"

META_4="$PROJECT_ROOT/data/test.json"                       # normal unsafes
DATASET_4="unsafes"
OUTDIR_4="$PROJECT_ROOT/results/normal"

# Build sampling flags passed through to VLGuard_eval.py
if [[ $MAX_QUESTIONS -gt 0 ]]; then
  echo "Will sample $MAX_QUESTIONS questions per dataset (seed=$SAMPLE_SEED)."
  SAMPLE_FLAGS="--max_questions $MAX_QUESTIONS --seed $SAMPLE_SEED"
else
  echo "Using all questions (no sampling)."
  SAMPLE_FLAGS=""
fi

# =====================
# Run inference: 4 cases sequentially on one GPU per model
# =====================
eval_model() {
  local gpu_id="$1"
  local engine_name="$2"
  local model_path="$3"

  echo "[GPU ${gpu_id}] START $engine_name"

  for case_args in \
    "$META_1|$DATASET_1|$OUTDIR_1" \
    "$META_2|$DATASET_2|$OUTDIR_2" \
    "$META_3|$DATASET_3|$OUTDIR_3" \
    "$META_4|$DATASET_4|$OUTDIR_4"
  do
    local meta="${case_args%%|*}"; local rest="${case_args#*|}"
    local dataset="${rest%%|*}"; local out_dir="${rest##*|}"

    mkdir -p "$out_dir/$dataset"
    echo "[GPU ${gpu_id}] dataset=$dataset meta=$meta"
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON_BIN" "$EVAL_PY" \
      --metaDir "$meta" \
      --imageDir "$IMAGE_DIR" \
      --dataset "$dataset" \
      --output_dir "$out_dir" \
      -e "$engine_name" \
      --model_path "$model_path" \
      $SAMPLE_FLAGS
  done

  echo "[GPU ${gpu_id}] DONE  $engine_name"
}

export -f eval_model
export PYTHON_BIN EVAL_PY IMAGE_DIR SAMPLE_FLAGS
export META_1 DATASET_1 OUTDIR_1
export META_2 DATASET_2 OUTDIR_2
export META_3 DATASET_3 OUTDIR_3
export META_4 DATASET_4 OUTDIR_4

# Launch up to 8 models in parallel (one per GPU), batch by GPU count
num_gpus=${#GPU_IDS[@]}
batch_start=0

while [[ $batch_start -lt ${#MODELS[@]} ]]; do
  pids=()
  for (( i=0; i<num_gpus && batch_start+i<${#MODELS[@]}; i++ )); do
    entry="${MODELS[$((batch_start + i))]}"
    ENGINE_NAME="${entry%%|*}"
    MODEL_PATH="${entry##*|}"
    GPU="${GPU_IDS[$i]}"

    echo ""
    echo "=============================="
    echo "Launching: $ENGINE_NAME  →  GPU $GPU"
    echo "=============================="
    eval_model "$GPU" "$ENGINE_NAME" "$MODEL_PATH" &
    pids+=($!)
  done

  for pid in "${pids[@]}"; do wait "$pid"; done
  (( batch_start += num_gpus ))
done

echo ""
echo "Phase 1 (model inference) completed."
