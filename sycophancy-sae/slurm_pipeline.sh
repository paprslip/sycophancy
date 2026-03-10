#!/bin/bash
#SBATCH --job-name=syco-sae
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --account=def-rgrosse
#SBATCH --time=3:00:00
#SBATCH --output=logs/%u-%x-%j.log
#SBATCH --error=logs/%u-%x-%j.log

# ══════════════════════════════════════════════════════════════════════════════
# Sycophancy SAE Pipeline — Single H100, all 5 steps
# Model:  Qwen3-14B-Instruct (subject)
# Judge:  Qwen3-4B-Instruct  (labeling)
# Est:    ~90 min on H100 80GB
#
# Usage:
#   sbatch scripts/slurm_pipeline.sh
#   sbatch --export=STEPS="1,2" scripts/slurm_pipeline.sh   # subset of steps
# ══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
CONFIG="configs/default.yaml"
STEPS="${STEPS:-1,2,3,4,5}"

export HF_HOME="/scratch/${USER}/hf_cache"
export HF_HUB_CACHE="${HF_HOME}"
export HF_TOKEN="$(cat ${HOME}/.cache/huggingface/token 2>/dev/null || true)"
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_XET=1
# export OPENAI_API_KEY=""    # uncomment if using OpenAI embeddings in steps 4/5

echo "══════════════════════════════════════════════════════"
echo "  Sycophancy SAE Pipeline"
echo "  Job:     ${SLURM_JOB_ID} on $(hostname)"
echo "  Started: $(date)"
echo "  Steps:   ${STEPS}"
echo "══════════════════════════════════════════════════════"

cd "${PROJECT_DIR}"
mkdir -p logs outputs/activations outputs/saes outputs/plots

# ── Create ephemeral venv on local SSD (CC recommended pattern) ───────────────
module load python/3.11 arrow
virtualenv --no-download "${SLURM_TMPDIR}/env"
source "${SLURM_TMPDIR}/env/bin/activate"
pip install --no-index --upgrade pip
pip install -r "${PROJECT_DIR}/requirements.txt"

export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"

echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo ""

# ── Step 1: Collect Activations ───────────────────────────────────────────────
if [[ "${STEPS}" == *"1"* ]]; then
    echo "── Step 1: Collecting Activations ──── $(date +%H:%M:%S)"
    python src/collect_activations.py --config "${CONFIG}"
    echo "   ✓ Done $(date +%H:%M:%S)"
fi

# ── Step 2: Train SAEs ────────────────────────────────────────────────────────
if [[ "${STEPS}" == *"2"* ]]; then
    echo "── Step 2: Training SAEs ────────────── $(date +%H:%M:%S)"
    python src/train_saes.py --config "${CONFIG}"
    echo "   ✓ Done $(date +%H:%M:%S)"
fi

# ── Step 3: Label Features ────────────────────────────────────────────────────
if [[ "${STEPS}" == *"3"* ]]; then
    echo "── Step 3: Labeling Features ─────────── $(date +%H:%M:%S)"
    # Subject model (Qwen3-14B) is unloaded after step 1 since collect_activations.py
    # exits. The judge model (Qwen3-4B) loads fresh here — fits easily in 80GB.
    python src/label_features.py --config "${CONFIG}"
    echo "   ✓ Done $(date +%H:%M:%S)"
fi

# ── Step 4: Evaluate Taxonomy ─────────────────────────────────────────────────
if [[ "${STEPS}" == *"4"* ]]; then
    echo "── Step 4: Evaluating Taxonomy ───────── $(date +%H:%M:%S)"
    python src/evaluate_taxonomy.py --config "${CONFIG}"
    echo "   ✓ Done $(date +%H:%M:%S)"
fi

# ── Step 5: Analyze Alignment ─────────────────────────────────────────────────
if [[ "${STEPS}" == *"5"* ]]; then
    echo "── Step 5: Analyzing Trigger Alignment ── $(date +%H:%M:%S)"
    python src/analyze_taxonomy_alignment.py --config "${CONFIG}"
    echo "   ✓ Done $(date +%H:%M:%S)"
fi

echo ""
echo "══════════════════════════════════════════════════════"
echo "  Pipeline complete: $(date)"
echo "  Results: ${PROJECT_DIR}/outputs/"
echo "══════════════════════════════════════════════════════"
