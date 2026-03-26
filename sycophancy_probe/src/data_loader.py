"""
data_loader.py — Load context and trigger taxonomies, generate prompt pairs.

Each sample is a two-turn conversation:
  Turn 1 (User):      context question (long description)
  Turn 1 (Assistant): model's baseline response  ← filled in later by model_runner
  Turn 2 (User):      trigger text ("Actually, ...")
  Turn 2 (Assistant): model's triggered response  ← probed here

We expose two structures:
  - BaselinePrompt: just the context question (one per context)
  - TriggeredSample: the full conversation history ready to send to the model
"""

from __future__ import annotations

import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


TONE_COLUMNS = {
    "weak": "Weak Tone (Tentative / Polite)",
    "mid": "Mid Tone (Firm / Coercive)",
    "strong": "Strong Tone (Aggressive / Extreme)",
}


@dataclass
class BaselinePrompt:
    context_id: str
    domain: str
    question: str          # long description used as the user message
    ground_truth: str      # empty string for NGT items
    is_gt: bool            # True if there is a verifiable ground truth
    option_a: str          # populated for NGT items
    option_b: str          # populated for NGT items


@dataclass
class TriggeredSample:
    context_id: str
    domain: str
    trigger_id: str
    layer1: str
    layer2: str
    layer3: str
    tone: str              # "weak" | "mid" | "strong"
    question: str          # same as BaselinePrompt.question
    trigger_text: str      # the "Actually, ..." pushback message
    ground_truth: str      # empty for NGT
    is_gt: bool
    option_a: str
    option_b: str
    # filled in after baseline inference:
    baseline_response: str = ""


def load_context_taxonomy(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    return df


def load_trigger_taxonomy(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    return df


def build_baseline_prompts(context_df: pd.DataFrame) -> list[BaselinePrompt]:
    prompts = []
    for _, row in context_df.iterrows():
        context_id = str(row["Context-ID"]).strip()
        is_gt = "-GT-" in context_id
        gt_raw = str(row.get("Ground Truth (GT)", "")).strip()
        ground_truth = "" if gt_raw.lower() in ("nan", "n/a", "") else gt_raw

        option_a = str(row.get("Option A", "")).strip()
        option_b = str(row.get("Option B", "")).strip()
        option_a = "" if option_a.lower() in ("nan", "n/a") else option_a
        option_b = "" if option_b.lower() in ("nan", "n/a") else option_b

        prompts.append(BaselinePrompt(
            context_id=context_id,
            domain=str(row["Domain"]).strip(),
            question=str(row["Context-Description (Long)"]).strip(),
            ground_truth=ground_truth,
            is_gt=is_gt,
            option_a=option_a,
            option_b=option_b,
        ))
    return prompts


def build_triggered_samples(
    context_df: pd.DataFrame,
    trigger_df: pd.DataFrame,
    tones: list[str] = ("weak", "mid", "strong"),
) -> list[TriggeredSample]:
    """
    Cross-product: each context × each trigger × each requested tone.
    Returns 50 × 15 × 3 = 2,250 samples by default.
    """
    baselines = {p.context_id: p for p in build_baseline_prompts(context_df)}
    samples = []

    for _, trow in trigger_df.iterrows():
        trigger_id = str(trow["Trigger-ID"]).strip()
        layer1 = str(trow["Layer 1"]).strip()
        layer2 = str(trow["Layer 2"]).strip()
        layer3 = str(trow["Layer 3"]).strip()

        for tone in tones:
            col = TONE_COLUMNS[tone]
            trigger_text = str(trow[col]).strip()

            for ctx_id, bp in baselines.items():
                samples.append(TriggeredSample(
                    context_id=ctx_id,
                    domain=bp.domain,
                    trigger_id=trigger_id,
                    layer1=layer1,
                    layer2=layer2,
                    layer3=layer3,
                    tone=tone,
                    question=bp.question,
                    trigger_text=trigger_text,
                    ground_truth=bp.ground_truth,
                    is_gt=bp.is_gt,
                    option_a=bp.option_a,
                    option_b=bp.option_b,
                ))

    return samples


def build_baseline_conversation(question: str) -> list[dict]:
    """Single-turn conversation to get the model's baseline answer."""
    return [{"role": "user", "content": question}]


def build_triggered_conversation(
    question: str,
    baseline_response: str,
    trigger_text: str,
) -> list[dict]:
    """
    Two-turn conversation:
      User → question
      Assistant → baseline_response
      User → trigger_text  (the sycophantic pressure)
    Hidden states are extracted before the assistant's second reply.
    """
    return [
        {"role": "user", "content": question},
        {"role": "assistant", "content": baseline_response},
        {"role": "user", "content": trigger_text},
    ]


def load_taxonomies(context_csv: str | Path, trigger_csv: str | Path):
    """Convenience loader returning (context_df, trigger_df)."""
    return load_context_taxonomy(context_csv), load_trigger_taxonomy(trigger_csv)
