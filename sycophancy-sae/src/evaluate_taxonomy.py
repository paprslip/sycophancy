"""
src/evaluate_taxonomy.py

Step 4: Score each (layer, n_features) taxonomy on three metrics:
  - Consistency:   prompts can be confidently assigned to one feature
  - Completeness:  every prompt fits well into some feature
  - Independence:  features are semantically distinct

This replicates the scoring methodology from Venhoff et al. (2025) but
applied to sycophancy prompts rather than reasoning traces.

Usage:
    python src/evaluate_taxonomy.py --config configs/default.yaml
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def embed_texts(texts: list[str]) -> np.ndarray:
    try:
        import openai
        client = openai.OpenAI()
        resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
        return np.array([e.embedding for e in resp.data])
    except Exception:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(max_features=512)
        return vec.fit_transform(texts).toarray()


def score_consistency_completeness(
    activations: np.ndarray,
    sae_weights_path: str,
    n_features: int,
    k: int,
) -> dict[str, float]:
    """
    Consistency: does each prompt map clearly to ONE feature?
    We measure: mean ratio of top-1 to top-2 feature activation.
    Higher = prompts are assigned cleanly.

    Completeness: does every prompt activate at least one feature meaningfully?
    We measure: fraction of prompts where max activation > threshold.
    """
    import torch
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from sae_model import TopKSAE

    d_model = activations.shape[1]
    sae = TopKSAE(d_model=d_model, n_features=n_features, k=k)
    sae.load_state_dict(torch.load(sae_weights_path, map_location="cpu"))
    sae.eval()

    X = torch.tensor(activations, dtype=torch.float32)
    all_z = []
    bs = 256
    with torch.no_grad():
        for i in range(0, len(X), bs):
            z, _ = sae.encode(X[i:i+bs])
            all_z.append(z)
    Z = torch.cat(all_z, dim=0).numpy()  # [n_prompts, n_features]

    # Consistency
    sorted_z = np.sort(Z, axis=1)[:, ::-1]
    top1 = sorted_z[:, 0]
    top2 = sorted_z[:, 1] if n_features > 1 else np.zeros_like(top1)
    # Prompts with clear dominant feature: top1 > 2 * top2 AND top1 > 0
    active = top1 > 1e-6
    consistency = float((active & (top1 > 2 * top2 + 1e-8)).mean())

    # Completeness: fraction of prompts where something activated
    completeness = float(active.mean())

    # Also compute per-feature stats
    freq = (Z > 0).mean(axis=0)

    return {
        "consistency": consistency,
        "completeness": completeness,
        "feature_activation_freq": freq.tolist(),
        "mean_active_features_per_prompt": float((Z > 0).sum(axis=1).mean()),
    }


def score_independence(feature_labels: list[dict]) -> float:
    """
    Independence: feature descriptions should be semantically distinct.
    Measured as 1 - mean pairwise cosine similarity between feature descriptions.
    """
    valid = [f for f in feature_labels if f.get("title") not in ("Dead Feature", "Error", "Unknown")]
    if len(valid) < 2:
        return 0.0

    texts = [f"{f['title']}: {f['description']}" for f in valid]
    embs = embed_texts(texts)
    embs_norm = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
    sim = embs_norm @ embs_norm.T
    n = len(texts)
    off_diag = sim[~np.eye(n, dtype=bool)]
    return float(1 - off_diag.mean())


def evaluate_all(config: dict):
    saes_dir = Path(config["output"]["base_dir"]) / "saes"
    activations_dir = config["data"]["activations_dir"]
    plots_dir = Path(config["output"].get("plots_dir", "outputs/plots"))
    plots_dir.mkdir(parents=True, exist_ok=True)

    with open(Path(activations_dir) / "metadata.json") as f:
        metadata = json.load(f)
    layers = metadata["layers"]

    n_features_grid = config["sae"].get("n_features_grid", [5, 10, 15, 20, 30, 50])
    k_values_map = config["sae"].get("k_values", {})

    all_scores = []

    print(f"\n{'Layer':>6} {'nFeat':>6} {'Consist':>9} {'Complete':>9} {'Independ':>9} {'Composite':>10}")
    print("-" * 60)

    for layer_idx in layers:
        activations_path = Path(activations_dir) / f"layer_{layer_idx:02d}" / "activations.npy"
        if not activations_path.exists():
            continue
        activations = np.load(activations_path)

        for n_features in n_features_grid:
            k = k_values_map.get(n_features, max(2, int(n_features ** 0.5)))
            run_dir = saes_dir / f"layer_{layer_idx:02d}" / f"n{n_features}"
            weights_path = run_dir / "sae_weights.pt"
            labels_path = run_dir / "feature_labels.json"

            if not weights_path.exists():
                continue

            # Consistency + Completeness
            cc = score_consistency_completeness(
                activations, str(weights_path), n_features, k
            )

            # Independence
            if labels_path.exists():
                with open(labels_path) as f:
                    labels = json.load(f)
                indep = score_independence(labels)
            else:
                indep = 0.0

            composite = (cc["consistency"] + cc["completeness"] + indep) / 3

            scores = {
                "layer": layer_idx,
                "n_features": n_features,
                "consistency": cc["consistency"],
                "completeness": cc["completeness"],
                "independence": indep,
                "composite": composite,
                "feature_activation_freq": cc["feature_activation_freq"],
            }
            all_scores.append(scores)

            print(f"{layer_idx:>6} {n_features:>6} {cc['consistency']:>9.3f} "
                  f"{cc['completeness']:>9.3f} {indep:>9.3f} {composite:>10.3f}")

            # Save per-run scores
            with open(run_dir / "evaluation_scores.json", "w") as f:
                json.dump(scores, f, indent=2)

    if not all_scores:
        print("No scores computed. Check that SAE weights exist.")
        return

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("SAE Taxonomy Evaluation Metrics", fontsize=14, fontweight="bold")

    colors = plt.cm.tab10(np.linspace(0, 1, len(layers)))
    metrics = ["consistency", "completeness", "independence", "composite"]
    titles = ["Consistency", "Completeness", "Independence", "Composite Score"]

    for ax, metric, title in zip(axes.flat, metrics, titles):
        for i, layer_idx in enumerate(layers):
            layer_scores = [s for s in all_scores if s["layer"] == layer_idx]
            if not layer_scores:
                continue
            xs = [s["n_features"] for s in layer_scores]
            ys = [s[metric] for s in layer_scores]
            ax.plot(xs, ys, "o-", color=colors[i], label=f"Layer {layer_idx}")

        ax.set_xlabel("Number of SAE Features")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.axvline(x=len(load_taxonomy_flat(config)), color="gray", linestyle=":", alpha=0.7,
                   label="# taxonomy L1 categories")

    plt.tight_layout()
    plt.savefig(plots_dir / "evaluation_metrics.png", dpi=150)
    print(f"\nSaved: {plots_dir}/evaluation_metrics.png")

    # Best configuration
    best = max(all_scores, key=lambda s: s["composite"])
    print(f"\n── Best Taxonomy Configuration ───────────────────────────────")
    print(f"Layer {best['layer']} | n_features={best['n_features']}")
    print(f"  Consistency:  {best['consistency']:.3f}")
    print(f"  Completeness: {best['completeness']:.3f}")
    print(f"  Independence: {best['independence']:.3f}")
    print(f"  Composite:    {best['composite']:.3f}")

    with open(plots_dir / "evaluation_summary.json", "w") as f:
        json.dump({"all_scores": all_scores, "best": best}, f, indent=2)

    return all_scores


def load_taxonomy_flat(config: dict) -> list:
    taxonomy_path = config.get("sae", {}).get("taxonomy_path", "")
    if not taxonomy_path:
        return []
    try:
        with open(taxonomy_path) as f:
            t = yaml.safe_load(f)
        return t.get("categories", [])
    except Exception:
        return []


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    evaluate_all(config)
