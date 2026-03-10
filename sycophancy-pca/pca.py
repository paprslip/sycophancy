import argparse
import csv
import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from jaxtyping import Float
from typing import Optional

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
import torch
import transformer_lens.utils as tl_utils
from transformer_lens import HookedTransformer
from pathlib import Path


@torch.inference_mode()
def _collect_prompt_activations(
    model: HookedTransformer,
    prompts: list[str],
    layer: int,
    hook_point: str,
    token_pos: int,
    batch_size: int,
) -> np.ndarray:
    act_name = tl_utils.get_act_name(hook_point, layer)
    rows = []
    tokenizer = model.tokenizer
    original_padding_side = getattr(tokenizer, "padding_side", None)

    if tokenizer is not None and hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = "left"

    try:
        for i in range(0, len(prompts), batch_size):
            tokens = model.to_tokens(prompts[i:i + batch_size]).to(model.cfg.device)
            _, cache = model.run_with_cache(tokens, names_filter=lambda name: name == act_name)
            activations = cache[hook_point, layer][:, token_pos, :].detach().cpu().numpy()
            rows.append(activations)
            del cache, tokens
    finally:
        if tokenizer is not None and original_padding_side is not None:
            tokenizer.padding_side = original_padding_side

    return np.concatenate(rows, axis=0)


def _labels_to_colors(labels: Optional[list[str]], n_points: int):
    if labels is None:
        return ["C0"] * n_points
    if len(labels) != n_points:
        raise ValueError("labels must have the same length as the number of points.")
    unique_labels = list(dict.fromkeys(labels))
    cmap = plt.cm.get_cmap("tab10", len(unique_labels))
    color_map = {label: cmap(i) for i, label in enumerate(unique_labels)}
    return [color_map[label] for label in labels], unique_labels, color_map

def pca(
    X: Optional[Float[np.ndarray, "batch d_model"]] = None,
    *,
    model: Optional[HookedTransformer] = None,
    prompts: Optional[list[str]] = None,
    layer: Optional[int] = None,
    hook_point: str = "resid_post",
    token_pos: int = -1,
    batch_size: int = 32,
    labels: Optional[list[str]] = None,
    n_components: int = 2,
    plot_3d: bool = False,
    annotate_indices: bool = True,
    title: str = "Feature Space Visualization",
):
    """
    Run PCA on either:
    1) A provided feature matrix X (shape: [n_samples, d_model]), or
    2) Activations collected from (model, prompts) at a specific hook/layer.
    """
    if X is None:
        if model is None or prompts is None or layer is None:
            raise ValueError(
                "Provide either X directly, or provide model, prompts, and layer."
            )
        if len(prompts) == 0:
            raise ValueError("prompts must be non-empty.")
        X = _collect_prompt_activations(
            model=model,
            prompts=prompts,
            layer=layer,
            hook_point=hook_point,
            token_pos=token_pos,
            batch_size=batch_size,
        )
    elif isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()

    if len(X) < 2:
        raise ValueError("Need at least 2 points for PCA.")
    if n_components < 2:
        raise ValueError("n_components must be >= 2.")
    if plot_3d and n_components < 3:
        raise ValueError("plot_3d=True requires n_components >= 3.")

    color_result = _labels_to_colors(labels, len(X))
    if labels is None:
        colors = color_result
        unique_labels = None
        color_map = None
    else:
        colors, unique_labels, color_map = color_result

    pca_model = PCA(n_components=n_components)
    Z = pca_model.fit_transform(X)

    if plot_3d:
        fig = plt.figure(figsize=(11, 8))
        ax = fig.add_subplot(111, projection="3d")
        marker_cycle = ["o", "^", "s", "D", "P", "X", "v", "<", ">", "*"]
        if unique_labels is None:
            ax.scatter(Z[:, 0], Z[:, 1], Z[:, 2], c=colors, alpha=0.85, s=80)
        else:
            for i, label in enumerate(unique_labels):
                idx = [j for j, cur_label in enumerate(labels) if cur_label == label]
                ax.scatter(
                    Z[idx, 0], Z[idx, 1], Z[idx, 2],
                    c=[color_map[label]] * len(idx),
                    marker=marker_cycle[i % len(marker_cycle)],
                    alpha=0.85, s=80, label=label
                )
            ax.legend()
        if annotate_indices:
            for i in range(len(Z)):
                ax.text(Z[i, 0], Z[i, 1], Z[i, 2], str(i), fontsize=8)
        ax.set_xlabel(f"PC1 ({pca_model.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca_model.explained_variance_ratio_[1]:.1%})")
        ax.set_zlabel(f"PC3 ({pca_model.explained_variance_ratio_[2]:.1%})")
        ax.set_title(title)
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(Z[:, 0], Z[:, 1], c=colors, alpha=0.7, s=80)
        if annotate_indices:
            for i in range(len(Z)):
                ax.annotate(str(i), (Z[i, 0], Z[i, 1]), xytext=(4, 4), textcoords="offset points", fontsize=8)

        ax.set_xlabel(f"PC1 ({pca_model.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca_model.explained_variance_ratio_[1]:.1%})")
        ax.set_title(title)
        if unique_labels is not None and color_map is not None:
            handles = [
                plt.Line2D([0], [0], marker="o", linestyle="", color=color_map[label], label=label)
                for label in unique_labels
            ]
            ax.legend(handles=handles)
        ax.grid(True, alpha=0.3)
    
    return fig, pca_model, Z


def pca_interactive_3d(
    Z: np.ndarray,
    pca_model: PCA,
    labels: Optional[list[str]] = None,
    title: str = "Feature Space Visualization",
    point_size: int = 5,
) -> go.Figure:
    """
    Create an interactive 3D Plotly scatter plot from PCA-projected coordinates.

    Args:
        Z: PCA-projected coordinates [n_samples, >=3]
        pca_model: fitted PCA model (for explained variance labels)
        labels: optional per-point category labels for coloring
        title: plot title
        point_size: marker size

    Returns:
        plotly Figure (call .write_html() to save)
    """
    if not HAS_PLOTLY:
        raise ImportError("plotly is required for interactive plots. Install with: pip install plotly")

    ev = pca_model.explained_variance_ratio_
    x_label = f"PC1 ({ev[0]:.1%})"
    y_label = f"PC2 ({ev[1]:.1%})"
    z_label = f"PC3 ({ev[2]:.1%})"

    fig = go.Figure()

    if labels is not None:
        unique_labels = list(dict.fromkeys(labels))
        colors = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        ]
        for i, label in enumerate(unique_labels):
            idx = [j for j, l in enumerate(labels) if l == label]
            fig.add_trace(go.Scatter3d(
                x=Z[idx, 0], y=Z[idx, 1], z=Z[idx, 2],
                mode="markers",
                name=label,
                marker=dict(size=point_size, color=colors[i % len(colors)], opacity=0.85),
                hovertemplate=(
                    f"<b>{label}</b><br>"
                    "idx: %{customdata}<br>"
                    "PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>PC3: %{z:.3f}"
                    "<extra></extra>"
                ),
                customdata=idx,
            ))
    else:
        fig.add_trace(go.Scatter3d(
            x=Z[:, 0], y=Z[:, 1], z=Z[:, 2],
            mode="markers",
            marker=dict(size=point_size, color="#1f77b4", opacity=0.85),
            hovertemplate="idx: %{customdata}<br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>PC3: %{z:.3f}<extra></extra>",
            customdata=list(range(len(Z))),
        ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=x_label,
            yaxis_title=y_label,
            zaxis_title=z_label,
        ),
        width=1000,
        height=800,
        legend=dict(itemsizing="constant"),
    )

    return fig


def load_prompt_records(
    dataset_path: Path,
    prompt_field: str = "prompt",
    label_field: Optional[str] = None,
    limit: Optional[int] = None,
) -> tuple[list[str], Optional[list[str]]]:
    suffix = dataset_path.suffix.lower()
    if suffix == ".json":
        with open(dataset_path, "r", encoding="utf-8") as f:
            records = json.load(f)
    elif suffix == ".csv":
        with open(dataset_path, newline="", encoding="utf-8") as f:
            records = list(csv.DictReader(f))
    else:
        raise ValueError("dataset_path must point to a .json or .csv file.")

    if not records:
        raise ValueError("Dataset is empty.")
    if prompt_field not in records[0]:
        raise ValueError(f"prompt field '{prompt_field}' not found in dataset.")
    if label_field is not None and label_field not in records[0]:
        raise ValueError(
            f"label field '{label_field}' not found in dataset. "
            f"Available fields: {list(records[0].keys())}"
        )

    if limit is not None:
        records = records[:limit]

    prompts = [str(r[prompt_field]) for r in records]
    labels = None if label_field is None else [str(r.get(label_field, "")) for r in records]
    return prompts, labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PCA over activations from sycophancy prompts.")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("data/sycophancy_prompts.json"),
        help="Path to prompt dataset (.json or .csv).",
    )
    parser.add_argument(
        "--prompt-field",
        type=str,
        default="prompt",
        help="Field/column name containing the prompt text.",
    )
    parser.add_argument(
        "--label-field",
        type=str,
        default="trigger_tone_type",
        help="Field/column name used for plot coloring/labels. Use '' to disable labels.",
    )
    parser.add_argument("--model-name", type=str, default="qwen3-1.7B")
    parser.add_argument("--layer", type=int, default=5)
    parser.add_argument("--start-layer", type=int, default=None, help="Start layer (inclusive).")
    parser.add_argument("--end-layer", type=int, default=None, help="End layer (inclusive).")
    parser.add_argument("--hook-point", type=str, default="resid_post")
    parser.add_argument("--token-pos", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-components", type=int, default=3)
    parser.add_argument("--plot-3d", action="store_true")
    parser.add_argument('--interactive', action='store_true', help='Save interactive 3D Plotly HTML alongside static PNGs (requires n_components >= 3).')
    parser.add_argument("--no-annotate-indices", action="store_true")
    parser.add_argument("--device", type=str, default=None, help="Model device, e.g. cuda:0, cuda, or cpu.")
    parser.add_argument("--save-dir", type=Path, default=Path("results/pca"), help="Directory to save PCA plots.")
    parser.add_argument("--no-show", action="store_true", help="Do not open interactive plot windows.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of prompts.")
    parser.add_argument("--title", type=str, default="Sycophancy Prompt Activations PCA")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    label_field = args.label_field if args.label_field else None
    prompts, labels = load_prompt_records(
        dataset_path=args.dataset_path,
        prompt_field=args.prompt_field,
        label_field=label_field,
        limit=args.limit,
    )

    if (args.start_layer is None) != (args.end_layer is None):
        raise ValueError("Provide both --start-layer and --end-layer, or neither.")
    if args.start_layer is None:
        layers = [args.layer]
    else:
        if args.end_layer < args.start_layer:
            raise ValueError("--end-layer must be >= --start-layer.")
        layers = list(range(args.start_layer, args.end_layer + 1))

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = HookedTransformer.from_pretrained(args.model_name, device=device)
    args.save_dir.mkdir(parents=True, exist_ok=True)

    if args.interactive and args.n_components < 3:
        print("Warning: --interactive requires n_components >= 3, setting n_components=3")
        args.n_components = 3

    variance_rows = []
    for layer in layers:
        fig, pca_model, Z = pca(
            model=model,
            prompts=prompts,
            layer=layer,
            hook_point=args.hook_point,
            token_pos=args.token_pos,
            batch_size=args.batch_size,
            labels=labels,
            n_components=args.n_components,
            plot_3d=args.plot_3d,
            annotate_indices=not args.no_annotate_indices,
            title=f"{args.title} | layer {layer}",
        )
        fig.tight_layout()
        out_path = args.save_dir / f"pca_layer_{layer}.png"
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Saved {out_path}")

        if args.interactive and Z.shape[1] >= 3:
            interactive_fig = pca_interactive_3d(
                Z, pca_model, labels=labels,
                title=f"{args.title} | layer {layer}",
            )
            html_path = args.save_dir / f"pca_layer_{layer}_interactive.html"
            interactive_fig.write_html(str(html_path), include_plotlyjs=True)
            print(f"Saved {html_path}")

        explained = pca_model.explained_variance_ratio_
        variance_rows.append(
            {
                "layer": layer,
                "pc1_explained_variance": float(explained[0]) if len(explained) > 0 else 0.0,
                "pc2_explained_variance": float(explained[1]) if len(explained) > 1 else 0.0,
                "pc3_explained_variance": float(explained[2]) if len(explained) > 2 else 0.0,
            }
        )
        print(f"  Explained variance: PC1={explained[0]:.1%} PC2={explained[1]:.1%} PC3={explained[2]:.1%}")

    variance_path = args.save_dir / "pca_explained_variance.csv"
    with open(variance_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "layer",
                "pc1_explained_variance",
                "pc2_explained_variance",
                "pc3_explained_variance",
            ],
        )
        writer.writeheader()
        writer.writerows(variance_rows)
    print(f"Saved {variance_path}")

    if not args.no_show:
        print("Plots were saved to disk; rerun with a single layer for interactive viewing if needed.")
