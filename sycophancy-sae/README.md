# Sycophancy SAE Taxonomy Discovery

Adapted from [thinking-llms-interp](https://github.com/cvenhoff/thinking-llms-interp) (Venhoff et al., 2025).

This pipeline trains **Top-K Sparse Autoencoders** (SAEs) on LLM activations from your sycophancy taxonomy prompts
to discover whether your taxonomy categories and layers are identifiable as features in the model's residual stream.

## Core Idea

- **2,250 sycophancy prompts** → run through a target LLM → extract residual stream activations per layer
- **Train SAEs with 5–50 features** (restricted decoder space forces SAE to find fundamental variance dimensions)
- **Subspace constraints** ensure features are orthogonal/structured according to your taxonomy hierarchy
- **LLM-based labeling** of each discovered feature → compare against your taxonomy categories
- **Evaluation metrics**: consistency, completeness, independence

## Pipeline Overview

```
prompts.jsonl
    │
    ▼
1. collect_activations.py   ← runs prompts through model, saves layer activations
    │
    ▼
2. train_saes.py            ← trains Top-K SAEs with subspace constraints for each (layer, n_features) combo
    │
    ▼
3. label_features.py        ← uses LLM to generate titles/descriptions for each SAE feature
    │
    ▼
4. evaluate_taxonomy.py     ← scores taxonomies on consistency/completeness/independence
    │
    ▼
5. analyze_taxonomy_alignment.py  ← compares discovered features to YOUR taxonomy categories
```

## Setup

```bash
pip install torch transformers datasets einops tqdm openai anthropic scikit-learn matplotlib seaborn pandas
```

## Quickstart

```bash
# 1. Prepare your prompts (see configs/prompts_format.md)
cp your_prompts.jsonl data/sycophancy_prompts.jsonl

# 2. Collect activations (picks layers at ~20%, 37%, 50%, 70% depth)
python scripts/run_pipeline.sh --model "meta-llama/Llama-3.1-8B-Instruct" --prompts data/sycophancy_prompts.jsonl

# Or run steps individually:
python src/collect_activations.py --config configs/default.yaml
python src/train_saes.py --config configs/default.yaml
python src/label_features.py --config configs/default.yaml
python src/evaluate_taxonomy.py --config configs/default.yaml
python src/analyze_taxonomy_alignment.py --config configs/default.yaml
```

## Key Design Choices (from the paper)

| Choice | Reasoning |
|---|---|
| Top-K sparsity (not L1) | Cleaner gradients, no shrinkage, exact control over active features |
| n_features ∈ [5, 50] | Forces SAE to find *fundamental* dimensions, not memorize |
| Restricted decoder (not overcomplete) | Identifies variance-explaining subspace components |
| Sentence/prompt-level activations | Captures semantic content, not token-level noise |
| Multiple layers | Concepts may be encoded at different depths |

## Subspace Constraints

Two types of subspace constraints are supported:

1. **Orthogonality constraint** – features must be approximately orthogonal (encourages monosemanticity)
2. **Hierarchical constraint** – features within the same taxonomy branch are constrained to a shared subspace; features across branches are pushed apart

See `src/constraints.py` for implementation.
