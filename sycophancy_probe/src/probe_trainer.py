"""
probe_trainer.py — Train and evaluate linear probes on per-layer hidden states.

For each transformer layer we train a logistic regression classifier that
predicts whether the model is in a "sycophantic" state (label=1) or not (label=0)
based on the hidden state vector at the last input token.

Output:
  - Per-layer accuracy and AUROC (via 5-fold stratified cross-validation)
  - Trained probe objects (saved as .pkl files)
  - Summary DataFrame and JSON
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def train_probes(
    hidden_states_per_layer: list[np.ndarray],  # list of [N, hidden_dim] arrays (one per layer)
    labels: np.ndarray,                          # [N] int array of 0/1
    n_folds: int = 5,
    output_dir: Optional[Path] = None,
    max_iter: int = 1000,
) -> pd.DataFrame:
    """
    Train a logistic regression probe per layer.

    Args:
        hidden_states_per_layer: List where index i is a [N, D] array of
            hidden states from layer i.
        labels: Binary labels [N], 0=non-sycophantic, 1=sycophantic.
        n_folds: Number of stratified CV folds.
        output_dir: If provided, save each probe as probes_layer_{i}.pkl.
        max_iter: Max iterations for LogisticRegression solver.

    Returns:
        DataFrame with columns: layer, accuracy_mean, accuracy_std, auroc_mean, auroc_std
    """
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    n_layers = len(hidden_states_per_layer)
    unique, counts = np.unique(labels, return_counts=True)
    logger.info(f"Training probes: {n_layers} layers, {len(labels)} samples, "
                f"label distribution: {dict(zip(unique.tolist(), counts.tolist()))}")

    if len(unique) < 2:
        raise ValueError(
            f"Labels must have at least 2 classes, got: {unique}. "
            "Check that some samples are labeled sycophantic and some are not."
        )

    records = []
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    for layer_idx, X in enumerate(hidden_states_per_layer):
        X = X.astype(np.float32)

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=max_iter,
                C=0.01,           # strong L2 regularization for high-dim/few-sample regime
                solver="liblinear",  # stable for binary classification, avoids lbfgs overflow
                random_state=42,
            )),
        ])

        acc_scores = cross_val_score(pipeline, X, labels, cv=skf, scoring="accuracy")
        auroc_scores = cross_val_score(pipeline, X, labels, cv=skf, scoring="roc_auc")

        record = {
            "layer": layer_idx,
            "accuracy_mean": float(acc_scores.mean()),
            "accuracy_std": float(acc_scores.std()),
            "auroc_mean": float(auroc_scores.mean()),
            "auroc_std": float(auroc_scores.std()),
        }
        records.append(record)

        logger.info(
            f"  Layer {layer_idx:3d}: acc={acc_scores.mean():.3f}±{acc_scores.std():.3f}  "
            f"auroc={auroc_scores.mean():.3f}±{auroc_scores.std():.3f}"
        )

        if output_dir is not None:
            # Fit on full data for the saved probe
            pipeline.fit(X, labels)
            probe_path = output_dir / f"probe_layer_{layer_idx:03d}.pkl"
            with open(probe_path, "wb") as f:
                pickle.dump(pipeline, f)

    df = pd.DataFrame(records)
    return df


def collect_hidden_states(
    labeled_samples,          # list of LabeledSample (from labeler.py)
    hidden_states_map: dict,  # {(context_id, trigger_id, tone): list[np.ndarray per layer]}
) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Align labeled samples with their hidden states and return per-layer matrices.

    Returns:
        (hidden_states_per_layer, labels)
        hidden_states_per_layer[i] has shape [N_valid, hidden_dim]
        labels has shape [N_valid]
    """
    valid_hs = []   # list of (list_of_layer_vecs, label)

    for sample in labeled_samples:
        if sample.label is None:
            continue  # skip unparseable samples
        key = (sample.context_id, sample.trigger_id, sample.tone)
        if key not in hidden_states_map:
            logger.warning(f"No hidden states for key {key}, skipping.")
            continue
        hs = hidden_states_map[key]  # list[np.ndarray] per layer
        valid_hs.append((hs, sample.label))

    if not valid_hs:
        raise ValueError("No valid labeled samples with hidden states found.")

    n_layers = len(valid_hs[0][0])
    labels = np.array([item[1] for item in valid_hs], dtype=np.int32)

    hidden_states_per_layer = []
    for layer_idx in range(n_layers):
        layer_vecs = np.stack([item[0][layer_idx] for item in valid_hs])
        hidden_states_per_layer.append(layer_vecs)

    return hidden_states_per_layer, labels


def save_metrics(metrics_df: pd.DataFrame, output_dir: Path) -> None:
    """Save metrics DataFrame as both CSV and JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    metrics_df.to_json(output_dir / "metrics.json", orient="records", indent=2)
    logger.info(f"Metrics saved to {output_dir}/metrics.{{csv,json}}")


def compute_breakdown_stats(
    labeled_samples,
    group_key: str,  # "domain" | "layer1" | "layer2" | "layer3" | "tone"
) -> pd.DataFrame:
    """
    Compute sycophancy rate (fraction of label=1 samples) grouped by a categorical key.
    Only includes samples where label is not None.
    """
    records = []
    for s in labeled_samples:
        if s.label is None:
            continue
        records.append({
            "domain": s.domain,
            "layer1": s.layer1,
            "layer2": s.layer2,
            "layer3": s.layer3,
            "tone": s.tone,
            "context_type": "GT (Factual)" if s.is_gt else "NGT (Opinion)",
            "label": s.label,
        })
    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame()
    grouped = df.groupby(group_key)["label"].agg(
        sycophancy_rate="mean",
        count="count",
    ).reset_index()
    return grouped
