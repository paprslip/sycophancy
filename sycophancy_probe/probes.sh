#!/bin/bash
#SBATCH --job-name=syco-probe-qwen3-14b
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --account=def-rgrosse
#SBATCH --time=2:00:00
#SBATCH --output=logs/%u-%x-%j.log
#SBATCH --error=logs/%u-%x-%j.log

set -euo pipefail

PROJECT_DIR="${SLURM_SUBMIT_DIR}"

export HF_HOME="/scratch/${USER}/hf_cache"
export HF_HUB_CACHE="${HF_HOME}"
export HF_TOKEN="$(cat ${HOME}/.cache/huggingface/token 2>/dev/null || true)"
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_XET=1

cd "${PROJECT_DIR}"

module load python/3.11 arrow
virtualenv --no-download "${SLURM_TMPDIR}/env"
source "${SLURM_TMPDIR}/env/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index -r "${PROJECT_DIR}/requirements.txt"

export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"

echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo ""

python main.py --model "Qwen/Qwen3-14B" --device cuda --batch-size 32 --max-new-tokens 512 --output-dir results/qwen3_14b/
