# Setup & Results Guide
# Qwen3-14B subject model · Qwen3-4B judge · def-rgrosse cluster

---

## 1. Model choices

### Subject model: Qwen3-14B-Instruct
Change from Llama-3.1-8B is already set in `configs/default.yaml`. Key differences:
- **Hidden dim**: 5120 (vs 4096 for 8B) — SAE input dimension is larger
- **Layers**: 40 (vs 32) — auto-selected layers become [8, 14, 20, 28]
- **dtype**: `bfloat16` — Qwen3 was trained in bfloat16; do NOT use float16 or you risk numerical issues
- **VRAM**: ~28GB in bfloat16, fits comfortably in H100 80GB
- **batch_size**: reduced to 8 (vs 16) to leave headroom

### Judge model: Qwen3-4B-Instruct
Used in Step 3 to label what each SAE feature represents.

Why Qwen3-4B specifically:
- Only ~8GB VRAM — loads easily on the same H100 after the subject model exits
- Same model family as subject, so it understands the prompt style
- Strong instruction following and JSON output compliance
- Free, no API key needed

Alternatives if you want to consider them:

| Judge | VRAM | Cost | JSON reliability | Notes |
|-------|------|------|-----------------|-------|
| **Qwen3-4B-Instruct** ← recommended | 8GB | Free | High | Same family as subject |
| Qwen3-8B-Instruct | 16GB | Free | High | Slightly better, still fits |
| claude-sonnet-4-20250514 | API | ~$0.10 | Very high | Best quality, costs money |
| gpt-4o-mini | API | ~$0.05 | High | Good quality, costs money |
| Qwen3-14B-Instruct | 28GB | Free | High | Overkill, ties up GPU longer |

If you have API credits and want the cleanest labels, use `claude-sonnet-4-20250514`
(set `provider: anthropic` in config and `export ANTHROPIC_API_KEY=...` in your job).
For a fully local/free run, Qwen3-4B is the right call.

---

## 2. First-time setup on the cluster

```bash
# Load modules (adjust names for your cluster — check with: module avail python)
module load python/3.11

# Create venv in your project directory
cd /path/to/sycophancy-sae
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Point HuggingFace cache at scratch storage (add to ~/.bashrc)
echo 'export HF_HOME=/scratch/$USER/hf_cache' >> ~/.bashrc
source ~/.bashrc

# Authenticate with HuggingFace (required for gated models like Qwen3)
huggingface-cli login

# Pre-download both models (do once interactively to avoid timeout in jobs)
srun --gres=gpu:h100:1 --mem=64G --time=0:30:00 --account=def-rgrosse --pty bash
source venv/bin/activate
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
for m in ['Qwen/Qwen3-14B-Instruct', 'Qwen/Qwen3-4B-Instruct']:
    print(f'Downloading {m}...')
    AutoTokenizer.from_pretrained(m)
    AutoModelForCausalLM.from_pretrained(m, torch_dtype=torch.bfloat16)
    print(f'  Done')
"
exit
```

Qwen3 models are gated — you must accept the license at https://huggingface.co/Qwen/Qwen3-14B-Instruct
and log in with `huggingface-cli login` before downloading.

---

## 3. Running the pipeline

### Option A: Single H100, all steps (~90 min)
```bash
sbatch scripts/slurm_pipeline.sh
```

### Option B: Parallelise across 4 H100s (~85 min, barely faster)
> The saving is small because Step 1 (activation collection) is sequential and takes ~25 min.
> Use Option A unless you have idle GPUs and want to experiment.

```bash
# Step 1: collect activations (1 GPU, ~25 min)
sbatch --export=STEPS="1" scripts/slurm_pipeline.sh

# Steps 2+3: train SAEs and label features, one layer per GPU (~45 min)
ARRAY_JOB=$(sbatch --parsable scripts/slurm_train_array.sh)
echo "Array job ID: $ARRAY_JOB"

# Steps 4+5: evaluate and align, runs after all array tasks finish (~15 min)
sbatch --dependency=afterok:${ARRAY_JOB} \
       --export=STEPS="4,5" \
       scripts/slurm_pipeline.sh
```

### Monitoring
```bash
squeue -u $USER                            # see job status
tail -f logs/$USER-syco-sae-<JOBID>.log   # watch live output
```

### Resuming after timeout
```bash
# If step 1 finished, skip it:
sbatch --export=STEPS="2,3,4,5" scripts/slurm_pipeline.sh
```

---

## 4. Expected timeline on 1x H100 80GB

| Step | What happens | Time |
|------|-------------|------|
| 1. collect_activations | 2,250 prompts through Qwen3-14B, extract 4 layers | ~25 min |
| 2. train_saes | 10 feature counts × 4 layers = 40 SAEs, 50 epochs each | ~15 min |
| 3. label_features | Load Qwen3-4B, label ~180 features × 3 reps | ~35 min |
| 4. evaluate_taxonomy | Consistency / completeness / independence | ~5 min |
| 5. analyze_alignment | Activation profiles + embedding alignment | ~10 min |
| **Total** | | **~90 min** |

---

## 5. Interpreting the results

All outputs land in `outputs/plots/`. Look at them in this order:

---

### Plot 1: `activation_profiles.png` — the primary diagnostic

Three heatmaps. The **left panel** is the key one:

```
         F0   F1   F2  ...  F14
T01       ■    ·    ·         ·     ← Uncertainty fires feature 0
T02       ·    ■    ·         ·     ← Authority fires feature 1
T03       ·    ·    ■         ·
...
T15       ·    ·    ·         ■     ← Future Consensus fires feature 14
```

**Diagonal = your trigger taxonomy is recoverable from Qwen3-14B's internals.**

The middle (Tone × Feature) and right (Domain × Feature) panels are confound checks.
If those are also diagonal, the SAE is responding to *how* prompts are phrased or
*what topic* they cover rather than the trigger mechanism — that's a problem.

**Ideal result:** left diagonal, middle+right uniform.

---

### Plot 2: `taxonomy_alignment_curves.png`

Coverage at each taxonomy level vs number of SAE features, one line per layer:

- **L3 coverage** (primary): fraction of 15 trigger types matched → want ~0.8+ at n=15
- **L2 coverage**: fraction of 3 mechanisms (Identity/Framing/Reward) → should reach 1.0 by n=5
- **L1 coverage**: fraction of 3 Jones categories → should reach 1.0 by n=3

The layer where L3 coverage peaks first (fewest features needed) is where your
taxonomy is most cleanly encoded. Layer ~37% depth (layer 14 for Qwen3-14B) is
the prediction from Venhoff et al.

---

### Plot 3: `feature_trigger_heatmap.png`

Each SAE feature's label (from Qwen3-4B) vs each trigger type by embedding similarity.
A good result has one bright cell per row, each in a different column.

---

### Key numbers to report

From `outputs/plots/alignment_summary.json`:
```json
{
  "best": {
    "layer": 14,
    "n_features": 15,
    "coverage_l3": 0.87,     // 13/15 trigger types recovered
    "coverage_l2": 1.0,      // all 3 L2 mechanisms found
    "coverage_l1": 1.0,      // all 3 Jones L1 categories found
    "precision": 0.73,       // 11/15 SAE features match a known trigger
    "selectivity": 0.61      // how distinctly triggers activate different features
  }
}
```

From `outputs/saes/layer_14/n15/feature_labels.json` — the actual feature labels:
```json
{"feature_idx": 3, "title": "Expert Authority Override",
 "trigger_layer_1": "Opinion Conformity",
 "trigger_layer_2": "Identity",
 "trigger_layer_3": "Expert Belief",    // ← matched T12?
 "confidence": "high"}
```

---

### If coverage is low (<0.5 at n=15)

Try in this order:

1. **Try layer 14** specifically — per Venhoff et al. the ~37% depth layer is most important
2. **Increase n_epochs to 100** in `configs/default.yaml`
3. **Switch pooling to `"mean"`** — the trigger is only the last 1-2 sentences; last-token
   may be dominated by the base question. Mean pooling weights the trigger more fairly.
4. **Extract trigger-only activations** — split the prompt at `"Actually,"` and run only
   the trigger suffix through the model (more surgical but requires prompt preprocessing)
5. **Try `subspace_constraint: "orthogonal"`** — hierarchical constraint may be
   over-constraining if your L1 categories aren't orthogonal in activation space
6. **Compare bfloat16 vs float32** for SAE training (activations are fine in bfloat16
   but SAE training sometimes benefits from float32 precision)
