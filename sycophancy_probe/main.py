#!/usr/bin/env python3
"""
main.py — Linear probe sycophancy detector.

Usage:
    python main.py --model "gpt2" --device cpu
    python main.py --model "meta-llama/Llama-3.2-1B-Instruct" --device mps --batch-size 4
    python main.py --model "mistralai/Mistral-7B-Instruct-v0.3" --device auto --batch-size 2 \\
                   --max-new-tokens 64 --output-dir results/mistral/

Pipeline:
  1. Load taxonomy CSVs
  2. Run baseline inference (no trigger) → get model's initial answers
  3. Run triggered inference (with sycophantic pressure) → get hidden states + triggered answers
  4. Label samples (sycophantic vs. not)
  5. Train linear probes per layer via cross-validation
  6. Save metrics, plots, and probe objects
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# ── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
logger = logging.getLogger("main")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train linear probes to detect sycophancy in HuggingFace models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", "-m",
        required=True,
        help="HuggingFace model name or local path (e.g. 'gpt2', 'meta-llama/Llama-3.2-1B-Instruct')",
    )
    parser.add_argument(
        "--context-csv",
        default="data/context_taxonomy.csv",
        help="Path to context taxonomy CSV",
    )
    parser.add_argument(
        "--trigger-csv",
        default="data/trigger_taxonomy.csv",
        help="Path to trigger taxonomy CSV",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="results",
        help="Directory to save probes, metrics, and plots",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda", "mps", "auto"],
        help="Device for model inference",
    )
    parser.add_argument(
        "--torch-dtype",
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Torch dtype for model weights",
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=1,
        help="Inference batch size",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Max tokens to generate per response",
    )
    parser.add_argument(
        "--tones",
        nargs="+",
        default=["weak", "mid", "strong"],
        choices=["weak", "mid", "strong"],
        help="Which trigger tones to include",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds for probe training",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of triggered samples (for fast testing; default: all)",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip generating matplotlib plots (useful in headless envs without display)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load taxonomy ──────────────────────────────────────────────────────
    logger.info("Loading taxonomy CSVs...")
    from src.data_loader import (
        load_taxonomies,
        build_baseline_prompts,
        build_triggered_samples,
    )
    context_df, trigger_df = load_taxonomies(args.context_csv, args.trigger_csv)
    baseline_prompts = build_baseline_prompts(context_df)
    triggered_samples = build_triggered_samples(context_df, trigger_df, tones=args.tones)

    if args.limit is not None:
        triggered_samples = triggered_samples[: args.limit]
        logger.info(f"Limiting to {args.limit} triggered samples (--limit flag)")

    logger.info(f"Baseline prompts: {len(baseline_prompts)}")
    logger.info(f"Triggered samples: {len(triggered_samples)}")

    # ── 2. Load model ─────────────────────────────────────────────────────────
    from src.model_runner import ModelRunner
    runner = ModelRunner(
        model_name=args.model,
        device=args.device,
        torch_dtype=args.torch_dtype,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )

    # ── 3. Baseline inference ─────────────────────────────────────────────────
    logger.info("Running baseline inference...")
    questions = [bp.question for bp in baseline_prompts]
    baseline_results = runner.run_baseline_batch(questions)

    # Map context_id → baseline response text
    baseline_response_map: dict[str, str] = {}
    for bp, result in zip(baseline_prompts, baseline_results):
        baseline_response_map[bp.context_id] = result.generated_text
        logger.debug(f"[{bp.context_id}] baseline: {result.generated_text!r}")

    # Attach baseline responses to triggered samples
    for ts in triggered_samples:
        ts.baseline_response = baseline_response_map.get(ts.context_id, "")

    # ── 4. Triggered inference ────────────────────────────────────────────────
    logger.info("Running triggered inference (extracting hidden states)...")

    hidden_states_map: dict[tuple, list[np.ndarray]] = {}
    labeled_samples = []

    from src.labeler import (
        label_gt_sample,
        label_ngt_sample,
        build_gt_expected_answers,
    )
    from src.data_loader import build_triggered_conversation

    gt_expected = build_gt_expected_answers(context_df)

    # Process in batches
    batch_size = args.batch_size
    for batch_start in tqdm(
        range(0, len(triggered_samples), batch_size),
        desc="Triggered inference",
        unit="batch",
    ):
        batch = triggered_samples[batch_start: batch_start + batch_size]

        questions_b = [ts.question for ts in batch]
        baseline_b = [ts.baseline_response for ts in batch]
        triggers_b = [ts.trigger_text for ts in batch]

        results = runner.run_triggered_batch(questions_b, baseline_b, triggers_b)

        for ts, result in zip(batch, results):
            key = (ts.context_id, ts.trigger_id, ts.tone)
            hidden_states_map[key] = result.hidden_states

            # Label
            if ts.is_gt:
                expected_tf = gt_expected.get(ts.context_id, False)
                sample = label_gt_sample(
                    ground_truth=ts.ground_truth,
                    expected_tf=expected_tf,
                    triggered_response=result.generated_text,
                    context_id=ts.context_id,
                    trigger_id=ts.trigger_id,
                    tone=ts.tone,
                    domain=ts.domain,
                    layer1=ts.layer1,
                    layer2=ts.layer2,
                    layer3=ts.layer3,
                    baseline_response=ts.baseline_response,
                )
            else:
                sample = label_ngt_sample(
                    baseline_response=ts.baseline_response,
                    triggered_response=result.generated_text,
                    context_id=ts.context_id,
                    trigger_id=ts.trigger_id,
                    tone=ts.tone,
                    domain=ts.domain,
                    layer1=ts.layer1,
                    layer2=ts.layer2,
                    layer3=ts.layer3,
                    ground_truth=ts.ground_truth,
                )
            labeled_samples.append(sample)

    # ── 5. Save raw labels ────────────────────────────────────────────────────
    import pandas as pd
    labels_records = [
        {
            "context_id": s.context_id,
            "trigger_id": s.trigger_id,
            "tone": s.tone,
            "domain": s.domain,
            "layer1": s.layer1,
            "layer2": s.layer2,
            "layer3": s.layer3,
            "is_gt": s.is_gt,
            "label": s.label,
            "baseline_parsed": s.baseline_parsed,
            "triggered_parsed": s.triggered_parsed,
            "baseline_response": s.baseline_response_raw,
            "triggered_response": s.triggered_response_raw,
        }
        for s in labeled_samples
    ]
    labels_df = pd.DataFrame(labels_records)
    labels_df.to_csv(output_dir / "labeled_samples.csv", index=False)
    logger.info(f"Saved {len(labeled_samples)} labeled samples → {output_dir}/labeled_samples.csv")

    n_valid = sum(1 for s in labeled_samples if s.label is not None)
    n_syco = sum(1 for s in labeled_samples if s.label == 1)
    logger.info(f"Valid samples: {n_valid} | Sycophantic: {n_syco} | Non-sycophantic: {n_valid - n_syco}")

    n_syco_count = sum(1 for s in labeled_samples if s.label == 1)
    n_nonsyco_count = n_valid - n_syco_count
    min_per_class = args.n_folds  # need at least n_folds samples per class for CV
    if n_valid < 2 * min_per_class or n_syco_count < min_per_class or n_nonsyco_count < min_per_class:
        logger.error(
            f"Not enough labeled samples for {args.n_folds}-fold CV. "
            f"Valid={n_valid}, sycophantic={n_syco_count}, non-sycophantic={n_nonsyco_count}. "
            "Try --limit with a larger value, or use an instruction-following model that outputs "
            "'True'/'False' or 'Option A'/'Option B'."
        )
        sys.exit(1)

    # ── 6. Collect hidden states and train probes ────────────────────────────
    logger.info("Collecting hidden states for probe training...")
    from src.probe_trainer import collect_hidden_states, train_probes, save_metrics

    hidden_states_per_layer, labels = collect_hidden_states(labeled_samples, hidden_states_map)
    logger.info(f"Hidden state matrix: {len(hidden_states_per_layer)} layers × {labels.shape[0]} samples × {hidden_states_per_layer[0].shape[1]} dims")

    logger.info(f"Training linear probes ({args.n_folds}-fold CV)...")
    metrics_df = train_probes(
        hidden_states_per_layer=hidden_states_per_layer,
        labels=labels,
        n_folds=args.n_folds,
        output_dir=output_dir,
        max_iter=1000,
    )

    save_metrics(metrics_df, output_dir)

    # ── 7. Visualize ──────────────────────────────────────────────────────────
    if not args.skip_plots:
        logger.info("Generating plots...")
        from src.visualizer import plot_probe_metrics, plot_all_breakdowns, print_summary
        plot_probe_metrics(metrics_df, output_dir)
        plot_all_breakdowns(labeled_samples, output_dir)
        print_summary(metrics_df, labeled_samples)
    else:
        logger.info("Skipping plots (--skip-plots)")
        from src.visualizer import print_summary
        print_summary(metrics_df, labeled_samples)

    # ── 8. Save run config ────────────────────────────────────────────────────
    config = vars(args)
    with open(output_dir / "run_config.json", "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"Done. All outputs saved to: {output_dir}/")


if __name__ == "__main__":
    main()
