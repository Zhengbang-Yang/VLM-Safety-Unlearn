#!/usr/bin/env bash
set -euo pipefail

# =====================
# Phase 2: Post-evaluation (rejection_eval + LLM judges)
# =====================
PYTHON_BIN="python"
PROJECT_ROOT="/egr/research-optml/chenyiw9/projects/VLGuard"
LLM_EVAL_DIR="$PROJECT_ROOT/llm_eval"
IMAGE_DIR="/egr/research-optml/chenyiw9/datasets/VLGuard_dataset/test"

# All 8 GPUs for LLM judges
GPU_IDS=(0 1 2 3 4 5 6 7)

# =====================
# Engine names to post-evaluate (must match the engine names from Phase 1)
# =====================
ENGINES=(
  # NPO (checkpoints-llava-retain-mix-npo-search-full-2)
  "llava-7b_npo_beta-0.8_retainA-1.0_lr-4.5e-7"
  "llava-7b_npo_beta-0.8_retainA-1.0_lr-4e-7"
  "llava-7b_npo_beta-0.85_retainA-1.0_lr-4.5e-7"
  "llava-7b_npo_beta-0.85_retainA-1.0_lr-4e-7"
  "llava-7b_npo_beta-0.85_retainA-1.0_lr-5e-7"
  "llava-7b_npo_beta-0.9_retainA-1.0_lr-4.5e-7"
  "llava-7b_npo_beta-0.9_retainA-1.0_lr-4e-7"
  "llava-7b_npo_beta-0.9_retainA-1.0_lr-5e-7"
  "llava-7b_npo_beta-0.95_retainA-1.0_lr-4.5e-7"
  "llava-7b_npo_beta-0.95_retainA-1.0_lr-4e-7"
  "llava-7b_npo_beta-0.95_retainA-1.0_lr-5e-7"
)

# =====================
# Result directories (must match Phase 1)
# =====================
OUTDIR_JB="$PROJECT_ROOT/results/jailbreak"    # jailbreak
OUTDIR_NM="$PROJECT_ROOT/results/normal"        # normal

# Instructions files for LLM judges
META_JB="$PROJECT_ROOT/safety_data/test_what_3shot.json"   # jailbreak unsafes
META_NM="$PROJECT_ROOT/data/test.json"                      # normal unsafes

# =====================
# Step 1: rejection_eval for all engines (CPU-only, run all in parallel)
# =====================
echo "Step 1: Running rejection_eval (CPU) for all engines ..."
pids=()
for ENGINE_NAME in "${ENGINES[@]}"; do
  LOG_REJ1="$OUTDIR_JB/safe_safes/${ENGINE_NAME}_rejection_eval.log"
  LOG_REJ3="$OUTDIR_NM/safe_safes/${ENGINE_NAME}_rejection_eval.log"

  "$PYTHON_BIN" "$LLM_EVAL_DIR/rejection_eval.py" \
    --outputs_file "$OUTDIR_JB/safe_safes/${ENGINE_NAME}.json" \
    > "$LOG_REJ1" 2>&1 &
  pids+=($!)

  "$PYTHON_BIN" "$LLM_EVAL_DIR/rejection_eval.py" \
    --outputs_file "$OUTDIR_NM/safe_safes/${ENGINE_NAME}.json" \
    > "$LOG_REJ3" 2>&1 &
  pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done
echo "Step 1 done."

# =====================
# Step 2: LLM judges — 24 GPU tasks (12 engines × 2 cases), batched 8 at a time
# (checkpoints-llava-retain-mix-npo-search-full)
#
#   Case 2: llm-judge-asr-3shot (jailbreak unsafes)
#   Case 4: llm-judge           (normal unsafes)
# =====================
echo ""
echo "Step 2: Running LLM judges across 8 GPUs ..."
mkdir -p "$OUTDIR_NM/unsafes/llm-eval"

# Build task list: "case|engine_name"
TASKS=()
for ENGINE_NAME in "${ENGINES[@]}"; do
  TASKS+=("case2|$ENGINE_NAME")
  TASKS+=("case4|$ENGINE_NAME")
done

# Run a single LLM judge task on a given GPU
run_judge_task() {
  local gpu_id="$1"
  local case_type="$2"
  local engine_name="$3"

  if [[ "$case_type" == "case2" ]]; then
    local log="$OUTDIR_JB/unsafes/${engine_name}_llm_judge.log"
    echo "[GPU ${gpu_id}] llm-judge-asr-3shot: $engine_name"
    CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON_BIN" "$LLM_EVAL_DIR/llm-judge-asr-3shot.py" \
      --outputs_file      "$OUTDIR_JB/unsafes/${engine_name}.json" \
      --instructions_file "$META_JB" \
      --base_image_path   "$IMAGE_DIR" \
      > "$log" 2>&1
  else
    local log="$OUTDIR_NM/unsafes/${engine_name}_llm_judge.log"
    echo "[GPU ${gpu_id}] llm-judge: $engine_name"
    CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON_BIN" "$LLM_EVAL_DIR/llm-judge.py" \
      --outputs_file           "$OUTDIR_NM/unsafes/${engine_name}.json" \
      --instructions_file      "$META_NM" \
      --base_image_path        "$IMAGE_DIR" \
      --relevant_output_file   "$OUTDIR_NM/unsafes/llm-eval/${engine_name}-relevant.json" \
      --irrelevant_output_file "$OUTDIR_NM/unsafes/llm-eval/${engine_name}-irrelevant.json" \
      > "$log" 2>&1
  fi
}

export -f run_judge_task
export PYTHON_BIN LLM_EVAL_DIR IMAGE_DIR
export OUTDIR_JB OUTDIR_NM META_JB META_NM

num_gpus=${#GPU_IDS[@]}
batch_start=0

while [[ $batch_start -lt ${#TASKS[@]} ]]; do
  pids=()
  for (( i=0; i<num_gpus && batch_start+i<${#TASKS[@]}; i++ )); do
    task="${TASKS[$((batch_start + i))]}"
    case_type="${task%%|*}"
    engine_name="${task##*|}"
    gpu="${GPU_IDS[$i]}"

    run_judge_task "$gpu" "$case_type" "$engine_name" &
    pids+=($!)
  done

  for pid in "${pids[@]}"; do wait "$pid"; done
  (( batch_start += num_gpus ))
done

echo ""
echo "All post-evaluations completed."
