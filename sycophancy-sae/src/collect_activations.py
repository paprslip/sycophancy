"""
src/collect_activations.py

Step 1: Run sycophancy prompts through a target LLM and save
residual stream activations at selected layers.

Usage:
    python src/collect_activations.py --config configs/default.yaml

Output:
    outputs/activations/{layer_idx}/activations.npy  - shape [n_prompts, d_model]
    outputs/activations/metadata.json               - prompt IDs, categories, etc.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_prompts(prompts_path: str, field_map: dict = None) -> list[dict]:
    """Load prompts from a JSON or JSONL file.

    Supports:
      - JSON array:  [ {...}, {...}, ... ]
      - JSONL:       one JSON object per line

    Uses field_map from config to normalise field names to the internal schema:
      id, prompt, category (trigger_layer_1), subcategory (trigger_layer_3), level (trigger_layer_2)

    All original fields are preserved under their original keys.
    """
    field_map = field_map or {}

    with open(prompts_path) as f:
        raw = f.read().strip()

    # Try JSON array first, fall back to JSONL
    try:
        records = json.loads(raw)
        if isinstance(records, dict):
            records = list(records.values())
    except json.JSONDecodeError:
        records = [json.loads(line) for line in raw.splitlines() if line.strip()]

    prompts = []
    for i, rec in enumerate(records):
        prompt = dict(rec)  # keep all original fields

        # Apply field_map to create canonical fields
        prompt["id"]          = rec.get(field_map.get("id", "id"),          rec.get("combo_id", str(i)))
        prompt["prompt"]      = rec.get(field_map.get("prompt", "prompt"),  rec.get("text", ""))
        prompt["category"]    = rec.get(field_map.get("category",    "category"),    rec.get("trigger_layer_1", ""))
        prompt["subcategory"] = rec.get(field_map.get("subcategory", "subcategory"), rec.get("trigger_layer_3", ""))
        prompt["level"]       = rec.get(field_map.get("level",       "level"),       rec.get("trigger_layer_2", ""))

        prompts.append(prompt)

    print(f"Loaded {len(prompts)} prompts from {prompts_path}")
    return prompts


def select_layers(model, layer_spec) -> list[int]:
    """
    Select which layers to extract from.

    layer_spec: "auto" or list of ints.

    "auto" picks layers at ~20%, 37%, 50%, 70% of model depth.
    37% is highlighted in Venhoff et al. as most causally important,
    but we recommend extracting multiple layers for comparison.
    """
    # Try to get total number of layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        n_layers = len(model.model.layers)
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        n_layers = len(model.transformer.h)
    else:
        # Fallback: count all children that look like transformer layers
        n_layers = sum(1 for _ in model.modules() if "Layer" in type(_).__name__)
        n_layers = max(n_layers, 32)  # reasonable default

    if layer_spec == "auto":
        fractions = [0.20, 0.37, 0.50, 0.70]
        layers = sorted(set(int(f * n_layers) for f in fractions))
        print(f"Auto-selected layers {layers} (out of {n_layers} total)")
        return layers
    else:
        return [int(l) for l in layer_spec]


def get_layer_hook(storage: dict, layer_idx: int, pooling: str):
    """Returns a forward hook that saves residual stream activations."""

    def hook(module, input, output):
        # output is typically (hidden_state, ...) or just hidden_state
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output

        # hidden: [batch, seq_len, d_model]
        if pooling == "last_token":
            vec = hidden[:, -1, :].detach().float().cpu()
        elif pooling == "mean":
            vec = hidden.mean(dim=1).detach().float().cpu()
        elif pooling == "max":
            vec = hidden.max(dim=1).values.detach().float().cpu()
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

        storage.setdefault(layer_idx, []).append(vec)

    return hook


def get_transformer_layer(model, layer_idx: int):
    """Get the transformer block at layer_idx."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer_idx]
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h[layer_idx]
    else:
        raise ValueError(
            "Could not find transformer layers. "
            "Please modify get_transformer_layer() for your model architecture."
        )


def collect_activations(
    config: dict,
    prompts_path: Optional[str] = None,
    output_dir: Optional[str] = None,
):
    cfg_model = config["model"]
    cfg_data = config["data"]

    prompts_path = prompts_path or cfg_data["prompts_path"]
    output_dir = output_dir or cfg_data["activations_dir"]
    pooling = cfg_data.get("pooling", "last_token")
    batch_size = cfg_model.get("batch_size", 16)

    os.makedirs(output_dir, exist_ok=True)

    # ── Load model ──────────────────────────────────────────────────────────
    model_name = cfg_model["name"]
    device = cfg_model.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}.get(
        cfg_model.get("dtype", "float32"), torch.float32
    )

    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {model_name} [{dtype}] → {device}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, trust_remote_code=True
    ).to(device)
    model.eval()

    # ── Select layers ───────────────────────────────────────────────────────
    layers = select_layers(model, cfg_model.get("layers", "auto"))

    # ── Register hooks ──────────────────────────────────────────────────────
    storage = {}
    hooks = []
    for layer_idx in layers:
        layer = get_transformer_layer(model, layer_idx)
        hook = layer.register_forward_hook(get_layer_hook(storage, layer_idx, pooling))
        hooks.append(hook)

    # ── Load prompts ─────────────────────────────────────────────────────────
    field_map = cfg_data.get("field_map", {})
    prompts = load_prompts(prompts_path, field_map)

    # ── Run forward passes ──────────────────────────────────────────────────
    print(f"Collecting activations for {len(prompts)} prompts across layers {layers}...")

    for batch_start in tqdm(range(0, len(prompts), batch_size)):
        batch = prompts[batch_start : batch_start + batch_size]
        texts = [p["prompt"] for p in batch]

        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        with torch.no_grad():
            model(**inputs)

    # ── Remove hooks ─────────────────────────────────────────────────────────
    for hook in hooks:
        hook.remove()

    # ── Save activations ─────────────────────────────────────────────────────
    for layer_idx, vecs in storage.items():
        layer_dir = Path(output_dir) / f"layer_{layer_idx:02d}"
        layer_dir.mkdir(parents=True, exist_ok=True)

        # Concatenate all batches → [n_prompts, d_model]
        activations = torch.cat(vecs, dim=0).numpy()
        np.save(layer_dir / "activations.npy", activations)
        print(f"  Layer {layer_idx}: saved {activations.shape}")

    # ── Save metadata ────────────────────────────────────────────────────────
    extra_fields = cfg_data.get("field_map", {}).get("extra_fields", [])
    metadata = {
        "model": model_name,
        "n_prompts": len(prompts),
        "layers": layers,
        "pooling": pooling,
        "prompts": [
            {
                "id":          p.get("id", str(i)),
                "category":    p.get("category", ""),       # trigger_layer_1
                "subcategory": p.get("subcategory", ""),    # trigger_layer_3 (15 leaf triggers)
                "level":       p.get("level", ""),          # trigger_layer_2
                **{f: p.get(f, "") for f in extra_fields},  # trigger_id, tone, domain, etc.
            }
            for i, p in enumerate(prompts)
        ],
    }
    with open(Path(output_dir) / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone! Activations saved to {output_dir}")
    return layers


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--prompts", default=None, help="Override prompts path")
    parser.add_argument("--output_dir", default=None, help="Override output dir")
    args = parser.parse_args()

    config = load_config(args.config)
    collect_activations(config, args.prompts, args.output_dir)
