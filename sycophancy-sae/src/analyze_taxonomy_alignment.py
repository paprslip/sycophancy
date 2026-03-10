"""
src/analyze_taxonomy_alignment.py

Step 5: Compare discovered SAE features to the Jones (1964) trigger taxonomy.

Primary question: do the 15 trigger types (T01–T15) appear as identifiable SAE features?
Secondary: does the L1 (Model Portrait / User Portrait / Opinion Conformity) and
           L2 (Identity / Framing / Reward) structure also emerge?

Also analyses:
  - Whether features cluster by tone (weak/mid/strong) rather than trigger type
  - Whether features cluster by domain (STEM/Health/etc.) rather than trigger type
  - Per-trigger activation profiles (which SAE features fire most for T01 vs T07 etc.)

Usage:
    python src/analyze_taxonomy_alignment.py --config configs/default.yaml
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import seaborn as sns
import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_taxonomy(path: str) -> dict:
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


def cosine_sim(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    return A @ B.T


def get_trigger_leaf_nodes(taxonomy: dict) -> list[dict]:
    """Flatten taxonomy to just the 15 leaf trigger nodes."""
    leaves = []
    for cat in taxonomy.get("categories", []):
        for sub in cat.get("subcategories", []):
            for leaf in sub.get("subcategories", []):
                leaves.append({
                    "trigger_id":      leaf["trigger_id"],
                    "trigger_layer_3": leaf["trigger_layer_3"],
                    "trigger_layer_2": sub["trigger_layer_2"],
                    "trigger_layer_1": cat["trigger_layer_1"],
                    "name":            leaf["name"],
                    "description":     leaf["description"],
                    "jones_mechanism": leaf.get("jones_mechanism", ""),
                })
    return leaves


def compute_label_alignment(
    feature_labels: list[dict],
    trigger_nodes: list[dict],
    sim_threshold: float = 0.55,
) -> dict:
    """
    For each SAE feature label, find the closest trigger type by embedding similarity.
    Returns alignment at all three taxonomy levels.
    """
    valid = [f for f in feature_labels if f.get("title") not in ("Dead Feature", "Error", "Unknown")]
    if not valid or not trigger_nodes:
        return {}

    feat_texts = [f"{f['title']}: {f['description']}" for f in valid]
    trig_texts = [f"{t['name']}: {t['description']} ({t['jones_mechanism']})" for t in trigger_nodes]

    all_embs  = embed_texts(feat_texts + trig_texts)
    feat_embs = all_embs[:len(feat_texts)]
    trig_embs = all_embs[len(feat_texts):]

    sim = cosine_sim(feat_embs, trig_embs)  # [n_features, 15]

    matches = []
    matched_l3 = set()
    matched_l2 = set()
    matched_l1 = set()

    for fi, feat in enumerate(valid):
        best_idx = int(np.argmax(sim[fi]))
        best_sim = float(sim[fi, best_idx])
        best_trig = trigger_nodes[best_idx]

        matched = best_sim >= sim_threshold
        if matched:
            matched_l3.add(best_trig["trigger_layer_3"])
            matched_l2.add(best_trig["trigger_layer_2"])
            matched_l1.add(best_trig["trigger_layer_1"])

        matches.append({
            "feature_idx":     feat.get("feature_idx"),
            "feature_title":   feat["title"],
            "predicted_l3":    best_trig["trigger_layer_3"],
            "predicted_l2":    best_trig["trigger_layer_2"],
            "predicted_l1":    best_trig["trigger_layer_1"],
            "predicted_tid":   best_trig["trigger_id"],
            "similarity":      best_sim,
            "matched":         matched,
            # Ground-truth from LLM labeling (if available)
            "labeled_l3":      feat.get("trigger_layer_3", ""),
            "labeled_l1":      feat.get("trigger_layer_1", ""),
        })

    n_valid = len(valid)
    n_l3_total = len(set(t["trigger_layer_3"] for t in trigger_nodes))
    n_l2_total = len(set(t["trigger_layer_2"] for t in trigger_nodes))
    n_l1_total = len(set(t["trigger_layer_1"] for t in trigger_nodes))

    return {
        "matches": matches,
        "coverage_l3": len(matched_l3) / n_l3_total,
        "coverage_l2": len(matched_l2) / n_l2_total,
        "coverage_l1": len(matched_l1) / n_l1_total,
        "precision":   sum(1 for m in matches if m["matched"]) / n_valid,
        "n_matched_l3": len(matched_l3),
        "n_matched_l2": len(matched_l2),
        "n_matched_l1": len(matched_l1),
        "sim_matrix":  sim.tolist(),
        "trigger_names": [t["trigger_layer_3"] for t in trigger_nodes],
        "feature_titles": [f["title"] for f in valid],
    }


def compute_activation_profile(
    metadata_prompts: list[dict],
    activations_path: str,
    sae_weights_path: str,
    n_features: int,
    k: int,
) -> dict:
    """
    Compute mean activation of each SAE feature per trigger_layer_3 category.
    This is the key empirical test: does each trigger type activate a different feature?

    Returns a dict with:
      trigger_feature_matrix: [n_triggers, n_features] mean activations
      trigger_names: list of trigger_layer_3 labels
    """
    try:
        import torch, sys
        sys.path.insert(0, str(Path(__file__).parent))
        from sae_model import TopKSAE

        acts = np.load(activations_path)
        d_model = acts.shape[1]
        sae = TopKSAE(d_model=d_model, n_features=n_features, k=k)
        sae.load_state_dict(torch.load(sae_weights_path, map_location="cpu"))
        sae.eval()

        X = torch.tensor(acts, dtype=torch.float32)
        all_z = []
        with torch.no_grad():
            for i in range(0, len(X), 256):
                z, _ = sae.encode(X[i:i+256])
                all_z.append(z)
        Z = torch.cat(all_z).numpy()  # [n_prompts, n_features]

        # Group by trigger_layer_3
        trigger_groups = defaultdict(list)
        for idx, p in enumerate(metadata_prompts):
            trigger_groups[p.get("subcategory", "unknown")].append(idx)

        trigger_names = sorted(trigger_groups.keys())
        matrix = np.zeros((len(trigger_names), n_features))
        for ti, tname in enumerate(trigger_names):
            idxs = trigger_groups[tname]
            matrix[ti] = Z[idxs].mean(axis=0)

        # Also compute by tone and domain for confound analysis
        tone_groups = defaultdict(list)
        domain_groups = defaultdict(list)
        for idx, p in enumerate(metadata_prompts):
            tone_groups[p.get("trigger_tone_type", "")].append(idx)
            domain_groups[p.get("context_domain", "")].append(idx)

        tone_names = sorted(tone_groups.keys())
        tone_matrix = np.zeros((len(tone_names), n_features))
        for ti, tname in enumerate(tone_names):
            idxs = tone_groups[tname]
            if idxs:
                tone_matrix[ti] = Z[idxs].mean(axis=0)

        domain_names = sorted(domain_groups.keys())
        domain_matrix = np.zeros((len(domain_names), n_features))
        for di, dname in enumerate(domain_names):
            idxs = domain_groups[dname]
            if idxs:
                domain_matrix[di] = Z[idxs].mean(axis=0)

        return {
            "trigger_feature_matrix": matrix.tolist(),
            "trigger_names": trigger_names,
            "tone_feature_matrix": tone_matrix.tolist(),
            "tone_names": tone_names,
            "domain_feature_matrix": domain_matrix.tolist(),
            "domain_names": domain_names,
        }
    except Exception as e:
        print(f"  Warning: could not compute activation profiles: {e}")
        return {}


def selectivity_score(matrix: np.ndarray) -> float:
    """
    Measure how selectively features respond to individual trigger types.
    High selectivity = each trigger activates a different feature.
    Uses mean max-to-sum ratio across features.
    """
    col_max = matrix.max(axis=0)
    col_sum = matrix.sum(axis=0) + 1e-8
    return float((col_max / col_sum).mean())


def run_alignment_analysis(config: dict):
    cfg_align   = config.get("alignment", {})
    taxonomy_path = cfg_align.get("taxonomy_path", config["sae"].get("taxonomy_path", ""))
    sim_threshold = cfg_align.get("similarity_threshold", 0.55)

    saes_dir  = Path(config["output"]["base_dir"]) / "saes"
    acts_dir  = Path(config["data"]["activations_dir"])
    plots_dir = Path(config["output"].get("plots_dir", "outputs/plots"))
    plots_dir.mkdir(parents=True, exist_ok=True)

    with open(acts_dir / "metadata.json") as f:
        metadata = json.load(f)
    layers = metadata["layers"]
    prompt_meta = metadata["prompts"]

    taxonomy = load_taxonomy(taxonomy_path) if taxonomy_path else {}
    trigger_nodes = get_trigger_leaf_nodes(taxonomy)
    print(f"Taxonomy: {len(trigger_nodes)} leaf trigger types")
    print(f"L1: {sorted(set(t['trigger_layer_1'] for t in trigger_nodes))}")
    print(f"L2: {sorted(set(t['trigger_layer_2'] for t in trigger_nodes))}")
    print(f"L3: {[t['trigger_layer_3'] for t in trigger_nodes]}")

    n_features_grid = config["sae"].get("n_features_grid", [5, 10, 15, 20, 30])
    k_values_map    = config["sae"].get("k_values", {})

    all_results = []

    print(f"\n{'Layer':>6} {'nFeat':>6} {'Cov-L3':>8} {'Cov-L2':>8} {'Cov-L1':>8} {'Prec':>6} {'Select':>8}")
    print("-" * 60)

    for layer_idx in layers:
        acts_path = acts_dir / f"layer_{layer_idx:02d}" / "activations.npy"
        if not acts_path.exists():
            continue

        for n_features in n_features_grid:
            run_dir    = saes_dir / f"layer_{layer_idx:02d}" / f"n{n_features}"
            labels_path = run_dir / "feature_labels.json"
            weights_path = run_dir / "sae_weights.pt"

            if not labels_path.exists():
                continue

            with open(labels_path) as f:
                labels = json.load(f)

            k = k_values_map.get(n_features, max(2, int(n_features ** 0.5)))

            # Label-based alignment
            alignment = compute_label_alignment(labels, trigger_nodes, sim_threshold)

            # Activation-profile-based analysis (empirical, doesn't need labeling)
            profile = {}
            if weights_path.exists():
                profile = compute_activation_profile(
                    prompt_meta, str(acts_path), str(weights_path), n_features, k
                )

            selectivity = 0.0
            if profile and "trigger_feature_matrix" in profile:
                M = np.array(profile["trigger_feature_matrix"])
                selectivity = selectivity_score(M)

            result = {
                "layer": layer_idx,
                "n_features": n_features,
                "coverage_l3": alignment.get("coverage_l3", 0),
                "coverage_l2": alignment.get("coverage_l2", 0),
                "coverage_l1": alignment.get("coverage_l1", 0),
                "precision":   alignment.get("precision", 0),
                "selectivity": selectivity,
                "matches":     alignment.get("matches", []),
                "profile":     profile,
            }
            all_results.append(result)

            print(f"{layer_idx:>6} {n_features:>6} "
                  f"{result['coverage_l3']:>8.2f} {result['coverage_l2']:>8.2f} "
                  f"{result['coverage_l1']:>8.2f} {result['precision']:>6.2f} "
                  f"{selectivity:>8.3f}")

            # Save per-run
            with open(run_dir / "taxonomy_alignment.json", "w") as f:
                json.dump({k: v for k, v in result.items() if k != "profile"}, f, indent=2)
            if profile:
                with open(run_dir / "activation_profiles.json", "w") as f:
                    json.dump(profile, f, indent=2)

    if not all_results:
        print("No results. Run label_features.py first.")
        return

    # ── Plots ─────────────────────────────────────────────────────────────────

    _plot_coverage_curves(all_results, layers, plots_dir)
    _plot_best_heatmap(all_results, trigger_nodes, plots_dir, saes_dir)
    _plot_activation_profiles(all_results, plots_dir)

    # Summary
    best = max(all_results, key=lambda r: r["coverage_l3"] + r["precision"] + r["selectivity"])
    print(f"\n── Best Configuration ────────────────────────────────────────────")
    print(f"Layer {best['layer']} | n_features={best['n_features']}")
    print(f"  L3 trigger coverage: {best['coverage_l3']:.0%}  ({best['coverage_l3']*15:.0f}/15 trigger types)")
    print(f"  L2 coverage:         {best['coverage_l2']:.0%}")
    print(f"  L1 coverage:         {best['coverage_l1']:.0%}")
    print(f"  Feature precision:   {best['precision']:.0%}")
    print(f"  Trigger selectivity: {best['selectivity']:.3f}")

    if best["matches"]:
        print(f"\n  Feature → Trigger matches:")
        for m in sorted(best["matches"], key=lambda x: -x["similarity"]):
            s = "✓" if m["matched"] else "✗"
            print(f"  {s} '{m['feature_title'][:35]:35s}' → {m['predicted_tid']} {m['predicted_l3']} (sim={m['similarity']:.2f})")

    with open(plots_dir / "alignment_summary.json", "w") as f:
        json.dump({"all_results": [{k: v for k, v in r.items() if k != "profile"} for r in all_results],
                   "best": {k: v for k, v in best.items() if k != "profile"}}, f, indent=2)

    return all_results, best


def _plot_coverage_curves(results, layers, plots_dir):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Trigger Taxonomy Alignment (Jones 1964)", fontsize=13, fontweight="bold")

    colors = cm.tab10(np.linspace(0, 1, len(layers)))
    metrics = [("coverage_l3", "L3 Trigger Type Coverage (15 types)"),
               ("coverage_l2", "L2 Mechanism Coverage (3 types)"),
               ("precision",   "Feature Precision")]

    for ax, (metric, title) in zip(axes, metrics):
        for i, layer_idx in enumerate(layers):
            lr = [r for r in results if r["layer"] == layer_idx]
            if not lr:
                continue
            ax.plot([r["n_features"] for r in lr],
                    [r[metric] for r in lr],
                    "o-", color=colors[i], label=f"Layer {layer_idx}")
        ax.set_xlabel("Number of SAE Features")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.05)
        if "L3" in title:
            ax.axvline(x=15, color="red", linestyle="--", alpha=0.4, label="n=15 triggers")

    plt.tight_layout()
    plt.savefig(plots_dir / "taxonomy_alignment_curves.png", dpi=150)
    print(f"Saved: {plots_dir}/taxonomy_alignment_curves.png")
    plt.close()


def _plot_best_heatmap(results, trigger_nodes, plots_dir, saes_dir):
    """Heatmap of SAE feature × trigger type similarity for best config."""
    best = max(results, key=lambda r: r["coverage_l3"] + r["precision"])
    matches = best.get("matches", [])
    if not matches:
        return

    trig_names = [t["trigger_layer_3"] for t in trigger_nodes]
    feat_titles = [m["feature_title"][:25] for m in matches]

    # Rebuild sim matrix from stored result
    align_path = saes_dir / f"layer_{best['layer']:02d}" / f"n{best['n_features']}" / "taxonomy_alignment.json"
    try:
        with open(align_path) as f:
            alignment = json.load(f)
        # Sim matrix isn't stored directly, so we rebuild from matches
        sim = np.zeros((len(matches), len(trig_names)))
        trig_idx = {t["trigger_layer_3"]: i for i, t in enumerate(trigger_nodes)}
        for fi, m in enumerate(matches):
            sim[fi, trig_idx.get(m["predicted_l3"], 0)] = m["similarity"]
    except Exception:
        return

    fig, ax = plt.subplots(figsize=(12, max(4, len(matches) * 0.4 + 1)))
    sns.heatmap(sim, ax=ax,
                xticklabels=[t[:15] for t in trig_names],
                yticklabels=feat_titles,
                annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1)
    ax.set_title(f"Feature → Trigger Alignment\nLayer {best['layer']}, n={best['n_features']}")
    ax.set_xlabel("Trigger Type (L3)")
    ax.set_ylabel("SAE Feature")
    plt.tight_layout()
    plt.savefig(plots_dir / "feature_trigger_heatmap.png", dpi=150)
    print(f"Saved: {plots_dir}/feature_trigger_heatmap.png")
    plt.close()


def _plot_activation_profiles(results, plots_dir):
    """
    For the best config: heatmap of trigger_type × feature mean activations.
    Key diagnostic: if triggers cluster by feature, the taxonomy is recovered.
    Also shows tone and domain activation profiles to check for confounds.
    """
    best = max(results, key=lambda r: r.get("selectivity", 0) + r.get("coverage_l3", 0))
    profile = best.get("profile", {})
    if not profile or "trigger_feature_matrix" not in profile:
        return

    M = np.array(profile["trigger_feature_matrix"])
    trig_names = profile["trigger_names"]
    n_features = M.shape[1]

    fig, axes = plt.subplots(1, 3, figsize=(18, max(5, len(trig_names) * 0.35 + 2)))
    fig.suptitle(f"Mean Feature Activations by Group\nLayer {best['layer']}, n={n_features}", fontsize=13)

    # 1. By trigger type
    sns.heatmap(M, ax=axes[0],
                yticklabels=trig_names,
                xticklabels=[f"F{i}" for i in range(n_features)],
                cmap="YlOrRd", annot=(n_features <= 20))
    axes[0].set_title("By Trigger Type (L3)\n← what we want to see")
    axes[0].set_xlabel("SAE Feature")

    # 2. By tone (confound check)
    if "tone_feature_matrix" in profile:
        T = np.array(profile["tone_feature_matrix"])
        sns.heatmap(T, ax=axes[1],
                    yticklabels=profile["tone_names"],
                    xticklabels=[f"F{i}" for i in range(n_features)],
                    cmap="YlOrRd")
        axes[1].set_title("By Tone (weak/mid/strong)\n← ideally uniform")
        axes[1].set_xlabel("SAE Feature")

    # 3. By domain (confound check)
    if "domain_feature_matrix" in profile:
        D = np.array(profile["domain_feature_matrix"])
        sns.heatmap(D, ax=axes[2],
                    yticklabels=profile["domain_names"],
                    xticklabels=[f"F{i}" for i in range(n_features)],
                    cmap="YlOrRd")
        axes[2].set_title("By Domain (STEM/Health/etc)\n← ideally uniform")
        axes[2].set_xlabel("SAE Feature")

    plt.tight_layout()
    plt.savefig(plots_dir / "activation_profiles.png", dpi=150)
    print(f"Saved: {plots_dir}/activation_profiles.png")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    run_alignment_analysis(config)
