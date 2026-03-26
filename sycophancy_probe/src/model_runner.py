"""
model_runner.py — HuggingFace model loading, inference, and hidden state extraction.

Key design:
- Supports any causal LM from HuggingFace (GPT-2, LLaMA, Mistral, Qwen, etc.)
- Uses output_hidden_states=True to capture all layer representations
- Extracts hidden state at the LAST INPUT TOKEN position (before generation begins)
  → this is the model's internal state when "deciding" how to respond
- Supports batched inference with automatic padding
- Supports device_map="auto" for large models
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    generated_text: str
    # hidden_states[i] is the float32 numpy array of shape [hidden_dim]
    # from layer i, at the last input token position
    hidden_states: list[np.ndarray]
    num_layers: int


class ModelRunner:
    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        torch_dtype: str = "auto",
        max_new_tokens: int = 64,
        batch_size: int = 1,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size

        logger.info(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"  # causal LM needs left-padding for batches

        dtype_map = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        resolved_dtype = dtype_map.get(torch_dtype, "auto")

        logger.info(f"Loading model: {model_name} on device={device}")
        if device == "auto":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=resolved_dtype,
                trust_remote_code=True,
            )
            self.device = next(self.model.parameters()).device
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=resolved_dtype,
                trust_remote_code=True,
            )
            self.device = torch.device(device)
            self.model = self.model.to(self.device)

        self.model.eval()
        logger.info(f"Model loaded. Layers: {self.model.config.num_hidden_layers}")

    def _format_messages(self, messages: list[dict]) -> str:
        """Apply chat template if available, else fall back to a simple format."""
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        # Simple fallback for models without a chat template (e.g. base GPT-2)
        text = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                text += f"User: {content}\n"
            elif role == "assistant":
                text += f"Assistant: {content}\n"
            elif role == "system":
                text += f"System: {content}\n"
        text += "Assistant: "
        return text

    @torch.no_grad()
    def run_single(self, messages: list[dict]) -> InferenceResult:
        """Run inference on a single conversation and extract hidden states."""
        prompt = self._format_messages(messages)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.device)

        input_len = inputs["input_ids"].shape[1]

        # Forward pass to get hidden states at last input token
        outputs = self.model(
            **inputs,
            output_hidden_states=True,
        )
        # outputs.hidden_states: tuple of (num_layers+1) tensors, each [1, seq_len, hidden_dim]
        # Index 0 = embedding layer; indices 1..N = transformer layers
        hidden_states = [
            hs[0, -1, :].float().cpu().numpy()  # last input token, all dims
            for hs in outputs.hidden_states
        ]

        # Generate the response
        gen_outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        new_tokens = gen_outputs[0][input_len:]
        generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        return InferenceResult(
            generated_text=generated_text,
            hidden_states=hidden_states,
            num_layers=len(hidden_states),
        )

    def _extract_hidden_states_single(self, prompt: str) -> list[np.ndarray]:
        """
        Extract hidden states for a single prompt with no padding.
        Causal LMs must not use padding in the forward pass for hidden state
        extraction — left-pad tokens corrupt activations by appearing in the
        attention context of real tokens.
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        return [hs[0, -1, :].float().cpu().numpy() for hs in outputs.hidden_states]

    @torch.no_grad()
    def run_batch(self, messages_list: list[list[dict]]) -> list[InferenceResult]:
        """
        Run inference on a batch of conversations.

        Hidden states are extracted per-sample (no padding) to avoid left-pad
        tokens corrupting the causal attention context.
        Text generation uses batching (with left-padding) for throughput.
        """
        results = []
        for i in range(0, len(messages_list), self.batch_size):
            chunk = messages_list[i: i + self.batch_size]
            prompts = [self._format_messages(m) for m in chunk]

            # ── Hidden states: per-sample, no padding ──────────────────────
            batch_hidden = [self._extract_hidden_states_single(p) for p in prompts]

            # ── Text generation: batched with left-padding ─────────────────
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(self.device)

            padded_input_len = inputs["input_ids"].shape[1]
            gen_outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            for sample_idx, hs_list in enumerate(batch_hidden):
                new_tokens = gen_outputs[sample_idx][padded_input_len:]
                text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                results.append(InferenceResult(
                    generated_text=text,
                    hidden_states=hs_list,
                    num_layers=len(hs_list),
                ))

        return results

    def run_baseline_batch(
        self, questions: list[str]
    ) -> list[InferenceResult]:
        """Convenience wrapper: run baseline (single-turn) inference on a list of questions."""
        from src.data_loader import build_baseline_conversation
        messages_list = [build_baseline_conversation(q) for q in questions]
        return self.run_batch(messages_list)

    def run_triggered_batch(
        self,
        questions: list[str],
        baseline_responses: list[str],
        trigger_texts: list[str],
    ) -> list[InferenceResult]:
        """Convenience wrapper: run triggered (two-turn) inference."""
        from src.data_loader import build_triggered_conversation
        messages_list = [
            build_triggered_conversation(q, br, tt)
            for q, br, tt in zip(questions, baseline_responses, trigger_texts)
        ]
        return self.run_batch(messages_list)
