# notebooks/explore_results.py
# Run as a Jupyter notebook or as a script with:
#   jupyter nbconvert --to notebook --execute notebooks/explore_results.py
#
# This notebook helps you interactively explore:
#   1. What each SAE feature learned
#   2. How features map to your taxonomy categories
#   3. Which layers/feature counts work best
#   4. Feature activation patterns across your 2,250 prompts

# %%
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sys.path.insert(0, str(Path("..") / "src"))

RESULTS_DIR = Path("../outputs")
SAES_DIR = RESULTS_DIR / "saes"
PLOTS_DIR = RESULTS_DIR / "plots"

# ── Load evaluation summary ──────────────────────────────────────────────────
# %%
with open(PLOTS_DIR / "evaluation_summary.json") as f:
    eval_summary = json.load(f)

scores = eval_summary["all_scores"]
best = eval_summary["best"]

print(f"Best configuration: Layer {best['layer']}, n_features={best['n_features']}")
print(f"Composite score: {best['composite']:.3f}")

# ── View all scores as a table ───────────────────────────────────────────────
# %%
import pandas as pd

df = pd.DataFrame(scores)
df_pivot = df.pivot(index="layer", columns="n_features", values="composite")
print("\nComposite scores (layer × n_features):")
print(df_pivot.round(3).to_string())

# ── Load and display feature labels for best config ──────────────────────────
# %%
best_dir = SAES_DIR / f"layer_{best['layer']:02d}" / f"n{best['n_features']}"
with open(best_dir / "feature_labels.json") as f:
    labels = json.load(f)

print(f"\nFeature labels for Layer {best['layer']}, n={best['n_features']}:")
print("-" * 70)
for feat in labels:
    status = "💀 DEAD" if feat["title"] == "Dead Feature" else feat["sycophancy_type"].upper()
    conf = feat.get("confidence", "?")
    print(f"[{feat['feature_idx']:2d}] {feat['title'][:40]:40s} | {status} | {conf}")
    print(f"     {feat['description'][:80]}")
    print()

# ── Load taxonomy alignment ──────────────────────────────────────────────────
# %%
with open(best_dir / "taxonomy_alignment.json") as f:
    alignment = json.load(f)

print(f"\nTaxonomy Alignment:")
print(f"Coverage:  {alignment['coverage']:.2f} ({alignment['n_matched_categories']}/{alignment['n_categories']} categories)")
print(f"Precision: {alignment['precision']:.2f} ({alignment['n_features_matched']}/{alignment['n_valid_features']} features matched)")

print("\nFeature → Category Matches:")
for m in sorted(alignment["match_details"], key=lambda x: -x["similarity"]):
    icon = "✓" if m["match"] else "✗"
    print(f"  {icon} '{m['feature_title'][:35]:35s}' → '{m['matched_category'][:30]:30s}' (sim={m['similarity']:.2f})")

# ── Heatmap: feature × taxonomy category similarity ──────────────────────────
# %%
sim_matrix = np.array(alignment["sim_matrix"])
cat_names = alignment["category_names"]
feat_titles = [m["feature_title"] for m in alignment["match_details"]]

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(
    sim_matrix,
    ax=ax,
    xticklabels=[c[:20] for c in cat_names],
    yticklabels=[t[:25] for t in feat_titles],
    annot=True, fmt=".2f",
    cmap="Blues", vmin=0, vmax=1,
)
ax.set_title(f"Feature-Taxonomy Similarity (Layer {best['layer']}, n={best['n_features']})")
ax.set_xlabel("Taxonomy Category")
ax.set_ylabel("SAE Feature")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "feature_taxonomy_heatmap.png", dpi=150)
plt.show()
print(f"Saved heatmap → {PLOTS_DIR}/feature_taxonomy_heatmap.png")

# ── Feature activation frequency across prompts ──────────────────────────────
# %%
with open(best_dir / "feature_stats.json") as f:
    stats = json.load(f)

freqs = np.array(stats["activation_frequency"])
feat_names = [feat["title"] for feat in labels]

fig, ax = plt.subplots(figsize=(10, 4))
bars = ax.bar(range(len(freqs)), freqs, color="steelblue", alpha=0.8)
ax.set_xticks(range(len(freqs)))
ax.set_xticklabels([n[:15] for n in feat_names], rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Fraction of Prompts Activating Feature")
ax.set_title("Feature Activation Frequency Across Sycophancy Prompts")
ax.axhline(1 / len(freqs), color="red", linestyle="--", alpha=0.5, label="Uniform baseline")
ax.legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / "feature_activation_freq.png", dpi=150)
plt.show()

# ── Per-category activation profiles ────────────────────────────────────────
# %%
# Load prompts metadata to see how features activate per category
import json

with open(RESULTS_DIR / "activations" / "metadata.json") as f:
    meta = json.load(f)

prompt_meta = meta["prompts"]  # list of {id, category, subcategory, level}
categories = sorted(set(p["category"] for p in prompt_meta if p["category"]))

print("\nTo see per-category feature activation profiles,")
print("load activations and run the SAE to get Z matrix, then group by category.")
print("(This requires the model weights to be available.)")

# ── Summary ──────────────────────────────────────────────────────────────────
# %%
print("\n" + "="*70)
print("SUMMARY: Are your taxonomy categories identifiable in the SAE?")
print("="*70)
print(f"Best layer:      {best['layer']}")
print(f"Best n_features: {best['n_features']}")
print(f"Coverage:        {alignment['coverage']:.0%} of your taxonomy L1 categories matched")
print(f"Precision:       {alignment['precision']:.0%} of SAE features align to your taxonomy")
print(f"Composite score: {best['composite']:.3f}")
print()
if alignment['coverage'] > 0.7:
    print("✓ HIGH COVERAGE: Your taxonomy categories are largely identifiable in the SAE features!")
elif alignment['coverage'] > 0.4:
    print("~ PARTIAL COVERAGE: Some taxonomy categories are identifiable, others may be too granular")
else:
    print("✗ LOW COVERAGE: The SAE may be finding different structure than your taxonomy")
    print("  Try: different layer, more features, or revising taxonomy categories")
