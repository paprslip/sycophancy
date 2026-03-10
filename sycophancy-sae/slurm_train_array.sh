#!/bin/bash
#SBATCH --job-name=syco-sae-train
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --account=def-rgrosse
#SBATCH --time=1:00:00
#SBATCH --output=logs/%u-%x-%A_%a.log
#SBATCH --error=logs/%u-%x-%A_%a.log
#SBATCH --array=0-3              # 4 tasks = 4 layers; each runs on its own H100

# ══════════════════════════════════════════════════════════════════════════════
# Parallel SAE Training + Labeling — SLURM Array Job
#
# Each array task handles ONE layer: trains all 10 SAE configs for that layer,
# then labels all features using Qwen3-4B-Instruct as judge.
#
# Prerequisite: Step 1 (collect_activations) must be done first.
#
# Workflow:
#   # 1. Collect activations (sequential, ~25 min)
#   sbatch --export=STEPS="1" scripts/slurm_pipeline.sh
#
#   # 2. Train + label in parallel across 4 H100s (~45 min, all layers at once)
#   ARRAY_JOB=$(sbatch --parsable scripts/slurm_train_array.sh)
#
#   # 3. Evaluate + align after all array tasks finish (~15 min)
#   sbatch --dependency=afterok:${ARRAY_JOB} \
#          --export=STEPS="4,5" scripts/slurm_pipeline.sh
#
# Total wall time: ~25 + 45 + 15 = ~85 min using 4 H100s vs ~90 min on 1 H100
# (The saving is modest — step 1 dominates and can't be parallelised easily)
# ══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
CONFIG="configs/default.yaml"

export HF_HOME="/scratch/${USER}/hf_cache"
export HF_HUB_CACHE="${HF_HOME}"
export HF_TOKEN="$(cat ${HOME}/.cache/huggingface/token 2>/dev/null || true)"
export HF_HUB_ENABLE_HF_TRANSFER=0
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"

cd "${PROJECT_DIR}"
mkdir -p logs outputs/saes

# ── Create ephemeral venv on local SSD (CC recommended pattern) ───────────────
module load python/3.11 arrow
virtualenv --no-download "${SLURM_TMPDIR}/env"
source "${SLURM_TMPDIR}/env/bin/activate"
pip install --no-index --upgrade pip
pip install -r "${PROJECT_DIR}/requirements.txt"

# Read the 4 layer indices from the activations metadata
LAYERS=($(python3 -c "
import json
with open('outputs/activations/metadata.json') as f:
    m = json.load(f)
print(' '.join(str(l) for l in sorted(m['layers'])))
"))

LAYER=${LAYERS[$SLURM_ARRAY_TASK_ID]}

echo "══════════════════════════════════════════════════════"
echo "  Array task ${SLURM_ARRAY_TASK_ID} → Layer ${LAYER}"
echo "  Node: $(hostname)"
echo "  GPU:  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "  Started: $(date)"
echo "══════════════════════════════════════════════════════"

# Train all SAE configs for this layer
echo "── Training SAEs for layer ${LAYER} ──── $(date +%H:%M:%S)"
python src/train_saes.py --config "${CONFIG}" --layer "${LAYER}"
echo "   ✓ Training done $(date +%H:%M:%S)"

# Label features for this layer using Qwen3-4B as judge
# Each GPU task loads its own copy of the judge model (4B * bfloat16 = ~8GB each — no problem)
echo "── Labeling features for layer ${LAYER} ── $(date +%H:%M:%S)"
python src/label_features.py --config "${CONFIG}" --layer "${LAYER}"
echo "   ✓ Labeling done $(date +%H:%M:%S)"

echo "Layer ${LAYER} complete: $(date)"
