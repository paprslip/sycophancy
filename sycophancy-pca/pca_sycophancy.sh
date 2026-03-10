#!/bin/bash
#SBATCH --job-name=pca_sycophancy
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --account=def-rgrosse
#SBATCH --time=4:00:00
#SBATCH --output=logs/%u-%x-%j.log
#SBATCH --error=logs/%u-%x-%j.log

set -euo pipefail

mkdir -p logs

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MPLBACKEND=Agg

export HF_HOME="/scratch/${USER}/hf_cache"
export HF_HUB_CACHE="${HF_HOME}"
export HF_TOKEN="$(cat ${HOME}/.cache/huggingface/token 2>/dev/null || true)"
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_XET=1

# ── Create ephemeral venv on local SSD (CC recommended pattern) ───────────────
module load python/3.11 arrow
virtualenv --no-download "${SLURM_TMPDIR}/env"
source "${SLURM_TMPDIR}/env/bin/activate"
pip install --no-index --upgrade pip
pip install -r /home/terrence/sycophancy/sycophancy-pca/requirements.txt

cd /home/terrence/sycophancy/sycophancy-pca/

# Defaults (override via env vars or by passing extra CLI args to sbatch script).
DATASET_PATH="${DATASET_PATH:-data/sycophancy_prompts.json}"
LABEL_FIELD="${LABEL_FIELD:-trigger_layer_1}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-14B}"
START_LAYER="${START_LAYER:-0}"
NUM_LAYERS="${NUM_LAYERS:-40}"
HOOK_POINT="${HOOK_POINT:-resid_post}"
BATCH_SIZE="${BATCH_SIZE:-32}"
N_COMPONENTS="${N_COMPONENTS:-3}"
TITLE="${TITLE:-Sycophancy Prompt Activations PCA}"
LIMIT="${LIMIT:-}"
SAVE_ROOT="${SAVE_ROOT:-results/pca_interactive}"
NUM_GPUS="${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-1}}"

END_LAYER=$((START_LAYER + NUM_LAYERS - 1))
echo "Running PCA for ${MODEL_NAME} across layers ${START_LAYER}..${END_LAYER} on ${NUM_GPUS} GPU(s)"

mkdir -p "${SAVE_ROOT}"
if (( NUM_GPUS > NUM_LAYERS )); then
  NUM_GPUS="${NUM_LAYERS}"
fi

BASE_CHUNK=$((NUM_LAYERS / NUM_GPUS))
REMAINDER=$((NUM_LAYERS % NUM_GPUS))
PIDS=()
CURRENT_START="${START_LAYER}"

for ((GPU=0; GPU<NUM_GPUS; GPU++)); do
  CHUNK_SIZE="${BASE_CHUNK}"
  if (( GPU < REMAINDER )); then
    CHUNK_SIZE=$((CHUNK_SIZE + 1))
  fi

  WORKER_START="${CURRENT_START}"
  WORKER_END=$((WORKER_START + CHUNK_SIZE - 1))
  CURRENT_START=$((WORKER_END + 1))

  SAVE_DIR="${SAVE_ROOT}/gpu_${GPU}"
  mkdir -p "${SAVE_DIR}"

  CMD=(
    python pca.py
    --dataset-path "${DATASET_PATH}"
    --prompt-field prompt
    --label-field "${LABEL_FIELD}"
    --model-name "${MODEL_NAME}"
    --start-layer "${WORKER_START}"
    --end-layer "${WORKER_END}"
    --hook-point "${HOOK_POINT}"
    --batch-size "${BATCH_SIZE}"
    --n-components "${N_COMPONENTS}"
    --plot-3d
    --interactive
    --device "cuda:0"
    --save-dir "${SAVE_DIR}"
    --no-show
    --title "${TITLE}"
  )

  if [[ -n "${LIMIT}" ]]; then
    CMD+=(--limit "${LIMIT}")
  fi

  # Forward any extra args given to this script directly to pca.py.
  CMD+=("$@")

  echo "GPU ${GPU}: layers ${WORKER_START}-${WORKER_END}"
  echo "Running: CUDA_VISIBLE_DEVICES=${GPU} ${CMD[*]}"
  CUDA_VISIBLE_DEVICES="${GPU}" "${CMD[@]}" &
  PIDS+=($!)
done

FAIL=0
for PID in "${PIDS[@]}"; do
  if ! wait "${PID}"; then
    FAIL=1
  fi
done

if (( FAIL != 0 )); then
  echo "One or more GPU workers failed."
  exit 1
fi

echo "All GPU workers completed successfully."
