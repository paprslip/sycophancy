"""
src/train_saes.py

Step 2: Train Top-K SAEs over a grid of (layer, n_features) configurations.

For each configuration, trains a SAE on the collected activations and saves:
  - Model weights
  - Training curves
  - Feature activation statistics

Usage:
    python src/train_saes.py --config configs/default.yaml
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from sae_model import SAETrainer, TopKSAE


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_activations(activations_dir: str, layer_idx: int) -> np.ndarray:
    path = Path(activations_dir) / f"layer_{layer_idx:02d}" / "activations.npy"
    if not path.exists():
        raise FileNotFoundError(f"No activations found at {path}")
    return np.load(path)


def load_metadata(activations_dir: str) -> dict:
    with open(Path(activations_dir) / "metadata.json") as f:
        return json.load(f)


def load_taxonomy_groups(taxonomy_path: str, n_features: int) -> Optional[list]:
    """
    Map taxonomy categories to feature indices for hierarchical constraint.

    Assigns SAE features to taxonomy groups proportionally.
    E.g., if you have 5 L1 categories and n_features=20, each group gets ~4 features.

    Returns list of lists, e.g. [[0,1,2,3], [4,5,6,7], ...]
    """
    if not taxonomy_path or not os.path.exists(taxonomy_path):
        return None

    with open(taxonomy_path) as f:
        taxonomy = yaml.safe_load(f)

    categories = taxonomy.get("categories", [])
    if not categories:
        return None

    n_groups = len(categories)
    # Distribute features across groups as evenly as possible
    base = n_features // n_groups
    extra = n_features % n_groups

    groups = []
    start = 0
    for i, cat in enumerate(categories):
        size = base + (1 if i < extra else 0)
        groups.append(list(range(start, start + size)))
        start += size

    print(f"  Hierarchical groups ({n_groups} categories → {n_features} features): "
          f"{[len(g) for g in groups]} features each")
    return groups


def normalize_activations(X: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Center and scale activations for stable SAE training."""
    mean = X.mean(axis=0)
    X_centered = X - mean
    scale = np.sqrt((X_centered ** 2).mean())
    X_norm = X_centered / (scale + 1e-8)
    return X_norm, scale, mean


def train_sae_for_config(
    activations: np.ndarray,
    n_features: int,
    k: int,
    config: dict,
    taxonomy_groups: Optional[list] = None,
    device: str = "cuda",
) -> tuple[TopKSAE, dict]:
    """
    Train a single SAE for a given (n_features, k) config.
    Returns trained SAE and training history dict.
    """
    cfg_sae = config["sae"]

    X_norm, scale, mean = normalize_activations(activations)
    X_tensor = torch.tensor(X_norm, dtype=torch.float32)

    d_model = activations.shape[1]
    n_epochs = cfg_sae.get("n_epochs", 50)
    lr = cfg_sae.get("learning_rate", 1e-3)
    aux_coeff = cfg_sae.get("aux_loss_coeff", 0.03)
    constraint_coeff = cfg_sae.get("constraint_loss_coeff", 0.1)
    constraint_type = cfg_sae.get("subspace_constraint", "orthogonal")
    batch_size = min(512, len(X_tensor))

    # Build SAE
    sae = TopKSAE(d_model=d_model, n_features=n_features, k=k).to(device)

    # Initialize pre-encoder bias to data mean
    with torch.no_grad():
        sae.b_pre.data = torch.tensor(X_norm.mean(0), dtype=torch.float32).to(device)

    trainer = SAETrainer(
        sae=sae,
        lr=lr,
        aux_loss_coeff=aux_coeff,
        constraint_loss_coeff=constraint_coeff,
        constraint_type=constraint_type,
        taxonomy_groups=taxonomy_groups,
    )

    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    history = {"loss_total": [], "loss_rec": [], "loss_con": [], "var_explained": []}

    for epoch in range(n_epochs):
        epoch_losses = {"loss_total": [], "loss_rec": [], "loss_con": []}

        for (batch,) in loader:
            batch = batch.to(device)
            losses = trainer.step(batch)
            for k_loss in epoch_losses:
                epoch_losses[k_loss].append(losses[k_loss])

        for k_loss in epoch_losses:
            history[k_loss].append(np.mean(epoch_losses[k_loss]))

        # Variance explained (on full dataset, every 10 epochs)
        if epoch % 10 == 0 or epoch == n_epochs - 1:
            sae.eval()
            with torch.no_grad():
                ve = trainer.variance_explained(X_tensor.to(device))
            history["var_explained"].append((epoch, ve))
            sae.train()

    return sae, history


def get_feature_stats(sae: TopKSAE, X: np.ndarray, device: str) -> dict:
    """Compute per-feature activation statistics over the full dataset."""
    sae.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32)
    all_z = []

    batch_size = 512
    for i in range(0, len(X_tensor), batch_size):
        batch = X_tensor[i : i + batch_size].to(device)
        with torch.no_grad():
            z, _ = sae.encode(batch)
        all_z.append(z.cpu())

    Z = torch.cat(all_z, dim=0).numpy()  # [n_prompts, n_features]

    return {
        "activation_frequency": (Z > 0).mean(axis=0).tolist(),  # fraction of prompts activating each feature
        "mean_activation": Z.mean(axis=0).tolist(),
        "max_activation": Z.max(axis=0).tolist(),
        "n_dead_features": int((Z.max(axis=0) < 1e-6).sum()),
    }


def run_grid_search(config: dict):
    cfg_sae = config["sae"]
    cfg_data = config["data"]

    activations_dir = cfg_data["activations_dir"]
    output_dir = Path(config["output"]["base_dir"]) / "saes"
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(activations_dir)
    layers = metadata["layers"]

    n_features_grid = cfg_sae.get("n_features_grid", [5, 10, 15, 20, 30, 50])
    k_values_map = cfg_sae.get("k_values", {})  # n_features → k

    taxonomy_path = cfg_sae.get("taxonomy_path", "")
    constraint_type = cfg_sae.get("subspace_constraint", "orthogonal")

    device = config["model"].get("device", "cuda" if torch.cuda.is_available() else "cpu")

    results_summary = []

    for layer_idx in layers:
        print(f"\n{'='*60}")
        print(f"Layer {layer_idx}")
        print(f"{'='*60}")

        activations = load_activations(activations_dir, layer_idx)
        X_norm, _, _ = normalize_activations(activations)
        print(f"  Activations shape: {activations.shape}")

        for n_features in n_features_grid:
            k = k_values_map.get(n_features, max(2, int(n_features ** 0.5)))
            print(f"\n  n_features={n_features}, k={k}")

            # Get taxonomy groups for hierarchical constraint
            taxonomy_groups = None
            if constraint_type == "hierarchical":
                taxonomy_groups = load_taxonomy_groups(taxonomy_path, n_features)

            # Train SAE
            sae, history = train_sae_for_config(
                X_norm, n_features, k, config, taxonomy_groups, device
            )

            # Get stats
            stats = get_feature_stats(sae, X_norm, device)
            final_ve = history["var_explained"][-1][1] if history["var_explained"] else 0.0
            final_loss = history["loss_total"][-1] if history["loss_total"] else float("nan")

            print(f"    VarExplained: {final_ve:.3f} | Final loss: {final_loss:.4f} | "
                  f"Dead features: {stats['n_dead_features']}/{n_features}")

            # Save
            run_dir = output_dir / f"layer_{layer_idx:02d}" / f"n{n_features}"
            run_dir.mkdir(parents=True, exist_ok=True)

            if config["output"].get("save_sae_weights", True):
                torch.save(sae.state_dict(), run_dir / "sae_weights.pt")

            with open(run_dir / "training_history.json", "w") as f:
                json.dump(history, f)

            with open(run_dir / "feature_stats.json", "w") as f:
                json.dump(stats, f, indent=2)

            config_info = {
                "layer": layer_idx,
                "n_features": n_features,
                "k": k,
                "d_model": activations.shape[1],
                "constraint_type": constraint_type,
                "var_explained": final_ve,
                "final_loss": final_loss,
                "n_dead_features": stats["n_dead_features"],
            }
            with open(run_dir / "config.json", "w") as f:
                json.dump(config_info, f, indent=2)

            results_summary.append(config_info)

    # Save overall summary
    with open(output_dir / "grid_search_summary.json", "w") as f:
        json.dump(results_summary, f, indent=2)

    print(f"\n\nGrid search complete! Results saved to {output_dir}")
    print_summary(results_summary)
    return results_summary


def print_summary(results: list[dict]):
    print("\n── Grid Search Summary ──────────────────────────────")
    print(f"{'Layer':>6} {'Features':>8} {'k':>3} {'VarExp':>8} {'Dead':>5}")
    print("-" * 40)
    for r in sorted(results, key=lambda x: (-x["layer"], x["n_features"])):
        print(f"{r['layer']:>6} {r['n_features']:>8} {r['k']:>3} "
              f"{r['var_explained']:>8.3f} {r['n_dead_features']:>5}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--layer", type=int, default=None,
                        help="Only train SAEs for this specific layer index. "
                             "Used by slurm_train_array.sh to parallelise across layers.")
    args = parser.parse_args()

    config = load_config(args.config)

    # If --layer is given, override the layers list to just that one layer.
    # This lets the SLURM array job run one layer per task in parallel.
    if args.layer is not None:
        meta_path = Path(config["data"]["activations_dir"]) / "metadata.json"
        with open(meta_path) as f:
            meta = json.load(f)
        if args.layer not in meta["layers"]:
            raise ValueError(f"Layer {args.layer} not found in activations metadata. "
                             f"Available layers: {meta['layers']}")
        # Override the module-level function so run_grid_search only sees one layer
        globals()["load_metadata"] = lambda path: {**meta, "layers": [args.layer]}

    run_grid_search(config)
