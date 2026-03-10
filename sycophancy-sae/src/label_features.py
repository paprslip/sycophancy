"""
src/label_features.py

Step 3: Use an LLM to generate human-interpretable labels for each SAE feature.

For each feature in each SAE, we:
  1. Find the top-N prompts most strongly activating that feature
  2. Sample N random prompts from the same cluster
  3. Show both sets to an LLM and ask for a title + description
  4. Repeat N times and take the most consistent label

Supported providers:
  - "anthropic"   → Claude via Anthropic API (needs ANTHROPIC_API_KEY)
  - "openai"      → GPT via OpenAI API (needs OPENAI_API_KEY)
  - "ollama"      → any locally hosted model via Ollama  (http://localhost:11434)
  - "local"       → any OpenAI-compatible local server   (vLLM, LM Studio, llama.cpp, etc.)

Usage:
    python src/label_features.py --config configs/default.yaml

    # Or override provider/model from the command line:
    python src/label_features.py --config configs/default.yaml --provider ollama --model llama3.1:8b
    python src/label_features.py --config configs/default.yaml --provider local  --model mistral --base_url http://localhost:8000/v1
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml

from sae_model import TopKSAE


# Jones (1964) trigger taxonomy — used to anchor the labeler
JONES_TAXONOMY_CONTEXT = """
The prompts come from a sycophancy trigger taxonomy grounded in Jones (1964) ingratiation theory.
Each prompt has a BASE QUESTION followed by a SYCOPHANCY TRIGGER appended at the end.
The trigger is the key part — it is the mechanism the user employs to pressure the model.

The taxonomy has three layers:
  Layer 1 (who is targeted):
    - Model Portrait: targets the model's identity, capabilities, or incentives
    - User Portrait: user presents their own identity/experience/emotion to pressure the model
    - Opinion Conformity: appeals to external consensus (experts, majority, established facts)

  Layer 2 (how pressure is applied):
    - Identity: appeals based on who someone is (expert, user's experience, model's self-image)
    - Framing: reframes the epistemic situation (established fact, prior agreement, limitation)
    - Reward: explicit incentive or threat contingent on agreement

  Layer 3 (specific trigger, 15 types):
    T01 Uncertainty        — model is made to doubt its answer
    T02 Authority          — human authority figure cited to override model
    T03 Positive Incentive — reward offered for agreement
    T04 Negative Incentive — punishment implied for disagreement
    T05 Prior Agreement    — model claimed to have agreed before
    T06 Limitation         — model's knowledge/training framed as biased/limited
    T07 Experience         — user's personal/vicarious experience cited
    T08 Emotional/Affective — user's emotional investment invoked
    T09 High Stakes        — personal consequences for user elevated
    T10 Certainty          — user expresses strong personal certainty
    T11 Self-Validation    — agreement framed as validating user's identity
    T12 Expert Belief      — expert opinion cited for opposite view
    T13 Popular Consensus  — majority/everyone cited for opposite view
    T14 Established Fact   — opposite framed as already-established fact
    T15 Future Consensus   — opposite framed as inevitable future consensus
"""

LABELING_PROMPT = """You are analyzing a Sparse Autoencoder (SAE) feature trained on LLM activations from sycophancy prompts.

Background on the prompt structure:
{taxonomy_context}

I will show you two sets of prompts. Each prompt has a BASE QUESTION and a SYCOPHANCY TRIGGER.
Focus entirely on the trigger text (the part after "Actually, ...") — that is what the SAE feature is responding to.

TOP ACTIVATING PROMPTS (highest SAE feature activation):
{top_prompts}

RANDOM PROMPTS FROM THE SAME CLUSTER:
{random_prompts}

Your task: identify WHICH TRIGGER MECHANISM most consistently appears in the top activating prompts.
Look at the trigger text (the sycophantic pressure added at the end) and identify the pattern.

Provide:
1. TITLE: A short (3-7 word) name for the trigger mechanism
2. DESCRIPTION: 2-3 sentences describing the specific pressure/manipulation tactic
3. TRIGGER_LAYER_1: Which Layer 1 category [Model Portrait / User Portrait / Opinion Conformity]
4. TRIGGER_LAYER_2: Which Layer 2 category [Identity / Framing / Reward]
5. TRIGGER_LAYER_3: Which specific trigger type from T01-T15 (or "mixed" if unclear)
6. CONFIDENCE: [low/medium/high]

Respond in this exact JSON format:
{{"title": "...", "description": "...", "trigger_layer_1": "...", "trigger_layer_2": "...", "trigger_layer_3": "...", "confidence": "..."}}
"""


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_prompts(prompts_path: str) -> list[dict]:
    with open(prompts_path) as f:
        raw = f.read().strip()

    # Support both JSON array and JSONL formats
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return list(data.values())
    except json.JSONDecodeError:
        pass

    # Fall back to JSONL (one JSON object per line)
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def load_sae(run_dir: Path, d_model: int, n_features: int, k: int) -> TopKSAE:
    sae = TopKSAE(d_model=d_model, n_features=n_features, k=k)
    weights_path = run_dir / "sae_weights.pt"
    sae.load_state_dict(torch.load(weights_path, map_location="cpu"))
    sae.eval()
    return sae


def get_feature_exemplars(
    sae: TopKSAE,
    activations: np.ndarray,
    prompts: list[dict],
    feature_idx: int,
    n_top: int = 50,
    n_random: int = 50,
) -> tuple[list[str], list[str]]:
    """Get top-activating and random prompts for a given feature.

    Uses the pre-TopK encoder output (with ReLU) rather than the post-TopK
    sparse code.  This avoids marking features as "dead" when k is small
    (e.g. k=1): the feature IS learned, it just rarely wins the top-k race.
    """
    X = torch.tensor(activations, dtype=torch.float32)
    with torch.no_grad():
        _, z_pre = sae.encode(X)

    # ReLU matches what TopK applies to the winners — negative pre-acts are inactive
    feature_acts = torch.relu(z_pre[:, feature_idx]).numpy()

    # Top-N activating
    top_idx = np.argsort(feature_acts)[::-1][:n_top]
    top_prompts = [prompts[i]["prompt"] for i in top_idx if feature_acts[i] > 0]

    # Random sample (from prompts that activate this feature at all)
    active_idx = np.where(feature_acts > 0)[0]
    if len(active_idx) > n_random:
        rand_idx = np.random.choice(active_idx, size=n_random, replace=False)
    else:
        rand_idx = active_idx
    random_prompts = [prompts[i]["prompt"] for i in rand_idx]

    return top_prompts, random_prompts


class HuggingFaceClient:
    """
    Wraps a locally loaded HuggingFace model so it can be used as a
    drop-in replacement for the API clients.

    The model is loaded once on construction and reused for all calls,
    so there is no per-call startup cost.

    Args:
        model_name_or_path: HuggingFace model ID or local directory path.
                            Any instruction-tuned model works well, e.g.:
                              "meta-llama/Llama-3.1-8B-Instruct"
                              "mistralai/Mistral-7B-Instruct-v0.3"
                              "Qwen/Qwen2.5-7B-Instruct"
                              "/path/to/your/local/model"
        device:    "auto" lets accelerate split across available GPUs/CPU.
                   Use "cuda:0" to pin to a specific GPU, or "cpu" to force CPU.
        dtype:     "float16" or "bfloat16" for GPU; "float32" for CPU.
        max_new_tokens: generation budget (512 is plenty for JSON labeling output).
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: str = "auto",
        dtype: str = "float16",
        max_new_tokens: int = 512,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        dtype_map = {
            "float16":  torch.float16,
            "bfloat16": torch.bfloat16,
            "float32":  torch.float32,
        }
        torch_dtype = dtype_map.get(dtype, torch.float16)

        print(f"  Loading HuggingFace model: {model_name_or_path} [{dtype}, device={device}]")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=device,          # "auto" shards across all available GPUs
            trust_remote_code=True,
        )
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        print(f"  Model ready.")

    def generate(self, prompt: str) -> str:
        """Run a single prompt and return the generated text (new tokens only)."""
        import torch

        # With device_map="auto" the model is sharded, so self.model.device
        # doesn't exist.  Resolve the device from the first parameter instead.
        try:
            device = self.model.device
        except AttributeError:
            device = next(self.model.parameters()).device

        # Use the model's chat template if it has one (most instruct models do)
        if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                add_generation_prompt=True,
            ).to(device)
        else:
            # Bare text fallback for models without a chat template
            input_ids = self.tokenizer(
                prompt, return_tensors="pt"
            ).input_ids.to(device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,            # greedy decoding → more consistent JSON
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Return only the newly generated tokens, not the echoed prompt
        new_tokens = output_ids[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)


def build_client(
    provider: str,
    base_url: Optional[str] = None,
    hf_model: Optional[str] = None,
    hf_device: str = "auto",
    hf_dtype: str = "float16",
):
    """
    Build the appropriate LLM client for the chosen provider.

      "anthropic"   - Anthropic API          (needs ANTHROPIC_API_KEY)
      "openai"      - OpenAI API             (needs OPENAI_API_KEY)
      "ollama"      - local Ollama server     (http://localhost:11434/v1)
      "local"       - OpenAI-compatible server (vLLM, LM Studio, llama.cpp)
      "huggingface" - load a HF model directly, no server needed
    """
    if provider == "anthropic":
        import anthropic
        return anthropic.Anthropic()

    elif provider == "openai":
        import openai
        return openai.OpenAI()

    elif provider == "ollama":
        import openai
        url = base_url or "http://localhost:11434/v1"
        return openai.OpenAI(base_url=url, api_key="ollama")

    elif provider == "local":
        import openai
        url = base_url or "http://localhost:8000/v1"
        return openai.OpenAI(base_url=url, api_key=os.environ.get("LOCAL_LLM_API_KEY", "local"))

    elif provider == "huggingface":
        if not hf_model:
            raise ValueError(
                "--model is required for provider=huggingface.\n"
                "Example: --model meta-llama/Llama-3.1-8B-Instruct\n"
                "         --model /path/to/local/model"
            )
        return HuggingFaceClient(hf_model, device=hf_device, dtype=hf_dtype)

    else:
        raise ValueError(
            f"Unknown provider: '{provider}'. "
            "Choose from: anthropic, openai, ollama, local, huggingface"
        )


def _extract_last_json_object(text: str) -> dict | None:
    """Find the last balanced {…} substring and try to parse it as JSON.

    Qwen3 thinking-mode output often looks like:
        'Okay, I need to look at {the triggers} and find...\n{"title": "...", ...}'
    A simple greedy regex would grab everything from the first '{' to the last '}',
    producing invalid JSON.  Instead, we scan backwards from the end of the text
    to find the outermost balanced braces of the LAST JSON object.
    """
    # Find the last '}' in the text
    end = text.rfind("}")
    if end == -1:
        return None

    # Walk backwards to find the matching '{'
    depth = 0
    for i in range(end, -1, -1):
        if text[i] == "}":
            depth += 1
        elif text[i] == "{":
            depth -= 1
        if depth == 0:
            candidate = text[i : end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                return None
    return None


def label_feature_with_llm(
    top_prompts: list[str],
    random_prompts: list[str],
    client,
    model: str,
    provider: str,
) -> dict:
    """
    Call the LLM to label a feature.
    Works with Anthropic, OpenAI, Ollama, local servers, and HuggingFace models.
    """
    import re

    top_text = "\n".join(f"[{i+1}] {p}" for i, p in enumerate(top_prompts[:20]))
    rand_text = "\n".join(f"[{i+1}] {p}" for i, p in enumerate(random_prompts[:10]))
    prompt = LABELING_PROMPT.format(
        taxonomy_context=JONES_TAXONOMY_CONTEXT,
        top_prompts=top_text,
        random_prompts=rand_text,
    )

    try:
        if provider == "anthropic":
            response = client.messages.create(
                model=model,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text

        elif provider == "huggingface":
            # HuggingFaceClient.generate() handles tokenization and decoding
            text = client.generate(prompt)

        else:
            # openai / ollama / local all share the OpenAI SDK interface
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=0.3,
            )
            text = response.choices[0].message.content

        # ── Strip thinking tags ─────────────────────────────────────────
        # Qwen3 models emit <think>…</think>.  skip_special_tokens=True
        # may strip the tags but leave the thinking *text* (with curly
        # braces that break naïve JSON regex).  Strip both forms.
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

        # ── Extract the LAST balanced JSON object ──────────────────────
        # Thinking text often contains stray { } that fool greedy regex.
        # Walk backward from the end to find the last '}', then scan
        # forward from its matching '{' to extract the real response JSON.
        parsed = _extract_last_json_object(text)
        if parsed is not None:
            return parsed

        return {
            "title": "Unknown",
            "description": text[:200],
            "sycophancy_type": "other",
            "confidence": "low",
        }

    except Exception as e:
        print(f"    LLM labeling error: {e!r}")
        print(f"    LLM raw output (first 300 chars): {text[:300] if 'text' in dir() else '(no text)'}")
        return {"title": "Error", "description": str(e), "trigger_layer_1": "", "trigger_layer_2": "", "trigger_layer_3": "other", "confidence": "low"}


def majority_vote_label(labels: list[dict]) -> dict:
    """Aggregate N labeling repetitions via majority vote on trigger taxonomy fields."""
    from collections import Counter

    titles = [l.get("title", "Unknown") for l in labels]
    best_title = Counter(titles).most_common(1)[0][0]

    # Vote on each taxonomy level independently
    l3s = [l.get("trigger_layer_3", l.get("sycophancy_type", "other")) for l in labels]
    l1s = [l.get("trigger_layer_1", "") for l in labels]
    l2s = [l.get("trigger_layer_2", "") for l in labels]
    best_l3 = Counter(l3s).most_common(1)[0][0]
    best_l1 = Counter(l1s).most_common(1)[0][0]
    best_l2 = Counter(l2s).most_common(1)[0][0]

    # Confidence based on agreement on the leaf trigger type
    if len(set(l3s)) == 1:
        confidence = "high"
    elif Counter(l3s).most_common(1)[0][1] > len(labels) // 2:
        confidence = "medium"
    else:
        confidence = "low"

    for l in labels:
        if l.get("title") == best_title:
            best_description = l.get("description", "")
            break
    else:
        best_description = labels[0].get("description", "")

    return {
        "title": best_title,
        "description": best_description,
        "trigger_layer_1": best_l1,
        "trigger_layer_2": best_l2,
        "trigger_layer_3": best_l3,
        "confidence": confidence,
        "all_titles": titles,
        "all_trigger_layer_3": l3s,
    }


def label_all_features(
    config: dict,
    provider_override: Optional[str] = None,
    model_override: Optional[str] = None,
    base_url_override: Optional[str] = None,
    hf_device: str = "auto",
    hf_dtype: str = "float16",
    layer_override: Optional[int] = None,   # set by --layer for SLURM array jobs
):
    cfg_labeling = config["labeling"]
    cfg_data = config["data"]
    cfg_sae = config["sae"]

    saes_dir = Path(config["output"]["base_dir"]) / "saes"
    activations_dir = cfg_data["activations_dir"]

    provider    = provider_override    or cfg_labeling.get("provider", "anthropic")
    model       = model_override       or cfg_labeling.get("model", "claude-sonnet-4-20250514")
    base_url    = base_url_override    or cfg_labeling.get("base_url", None)
    hf_device   = cfg_labeling.get("hf_device", hf_device)
    hf_dtype    = cfg_labeling.get("hf_dtype",  hf_dtype)
    n_exemplars = cfg_labeling.get("n_exemplars", 50)
    n_random    = cfg_labeling.get("n_random", 50)
    n_reps      = cfg_labeling.get("n_repetitions", 3)

    print(f"Provider: {provider} | Model: {model}"
          + (f" | base_url: {base_url}" if base_url else ""))

    client = build_client(provider, base_url, hf_model=model,
                          hf_device=hf_device, hf_dtype=hf_dtype)

    # Load prompts
    prompts = load_prompts(cfg_data["prompts_path"])

    # Load metadata
    with open(Path(activations_dir) / "metadata.json") as f:
        metadata = json.load(f)
    layers = metadata["layers"]
    if layer_override is not None:
        if layer_override not in layers:
            raise ValueError(f"Layer {layer_override} not in activations metadata. Available: {layers}")
        layers = [layer_override]
        print(f"  (Restricted to layer {layer_override} by --layer flag)")

    n_features_grid = cfg_sae.get("n_features_grid", [5, 10, 15, 20, 30, 50])
    k_values_map = cfg_sae.get("k_values", {})

    for layer_idx in layers:
        activations_path = Path(activations_dir) / f"layer_{layer_idx:02d}" / "activations.npy"
        activations = np.load(activations_path)

        for n_features in n_features_grid:
            run_dir = saes_dir / f"layer_{layer_idx:02d}" / f"n{n_features}"
            weights_path = run_dir / "sae_weights.pt"

            if not weights_path.exists():
                print(f"  Skipping layer {layer_idx} / n{n_features} (no weights)")
                continue

            labels_path = run_dir / "feature_labels.json"
            if labels_path.exists():
                print(f"  Skipping layer {layer_idx} / n{n_features} (already labeled)")
                continue

            print(f"\nLabeling layer {layer_idx} / n{n_features}...")

            k = k_values_map.get(n_features, max(2, int(n_features ** 0.5)))
            d_model = activations.shape[1]
            sae = load_sae(run_dir, d_model, n_features, k)

            feature_labels = []
            for feat_idx in range(n_features):
                top_prompts, rand_prompts = get_feature_exemplars(
                    sae, activations, prompts, feat_idx, n_exemplars, n_random
                )

                if len(top_prompts) == 0:
                    feature_labels.append({
                        "feature_idx": feat_idx,
                        "title": "Dead Feature",
                        "description": "This feature never activated on any prompt.",
                        "sycophancy_type": "other",
                        "confidence": "low",
                    })
                    continue

                reps = []
                for rep in range(n_reps):
                    label = label_feature_with_llm(top_prompts, rand_prompts, client, model, provider)
                    reps.append(label)

                final_label = majority_vote_label(reps)
                final_label["feature_idx"] = feat_idx

                print(f"  Feature {feat_idx:2d}: '{final_label['title']}' "
                      f"[{final_label['trigger_layer_3']}] ({final_label['confidence']})")
                feature_labels.append(final_label)

            with open(labels_path, "w") as f:
                json.dump(feature_labels, f, indent=2)

    print("\nLabeling complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--provider", default=None,
                        choices=["anthropic", "openai", "ollama", "local", "huggingface"],
                        help="Override provider from config")
    parser.add_argument("--model", default=None,
                        help=(
                            "Model name or path. Examples:\n"
                            "  anthropic:   claude-sonnet-4-20250514\n"
                            "  openai:      gpt-4o-mini\n"
                            "  ollama:      llama3.1:8b\n"
                            "  huggingface: meta-llama/Llama-3.1-8B-Instruct\n"
                            "               /path/to/local/model"
                        ))
    parser.add_argument("--base_url", default=None,
                        help="Base URL for local server (e.g. http://localhost:11434/v1)")
    parser.add_argument("--hf_device", default="auto",
                        help="Device for HuggingFace model: 'auto', 'cuda', 'cpu', 'cuda:0'")
    parser.add_argument("--hf_dtype", default="float16",
                        choices=["float16", "bfloat16", "float32"],
                        help="dtype for HuggingFace model (default: float16)")
    parser.add_argument("--layer", type=int, default=None,
                        help="Only label features for this specific layer index (used by SLURM array job)")
    args = parser.parse_args()

    config = load_config(args.config)
    label_all_features(
        config,
        provider_override=args.provider,
        model_override=args.model,
        base_url_override=args.base_url,
        hf_device=args.hf_device,
        hf_dtype=args.hf_dtype,
        layer_override=args.layer,
    )
