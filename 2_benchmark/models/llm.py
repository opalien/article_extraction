"""Lightweight interface to run a causal LLM for question answering over long articles."""

from __future__ import annotations

import os
import textwrap
from typing import Dict, Tuple

import torch
from huggingface_hub.errors import GatedRepoError
from requests import HTTPError
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers.pipelines import TextGenerationPipeline

MODEL_ID = os.environ.get("BENCHMARK_LLM_MODEL_ID", "gemma3:4b")
DEFAULT_CONTEXT_TOKENS = 3072
MAX_NEW_TOKENS = 128
PROMPT_OVERHEAD_TOKENS = 256
FALLBACK_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Map friendly aliases to actual Hugging Face repo ids.
MODEL_ALIASES = {
    "gemma3:4b": "google/gemma-2-2b-it",
}

generators: Dict[str, TextGenerationPipeline] = {}
failed_generators: Dict[str, Exception] = {}


def _resolve_model_id(model_id: str) -> Tuple[str, bool]:
    """Return a valid Hugging Face repo id along with a flag indicating alias usage."""
    if model_id in MODEL_ALIASES:
        return MODEL_ALIASES[model_id], True

    resolved = model_id
    if ":" in resolved:
        resolved = resolved.replace(":", "-")
    return resolved, resolved != model_id


def _load_generator(model_id: str):
    """Lazily load and cache a text-generation pipeline for the requested model."""
    resolved_id, was_alias = _resolve_model_id(model_id)

    for candidate_id in [resolved_id, FALLBACK_MODEL_ID if resolved_id != FALLBACK_MODEL_ID and model_id == MODEL_ID else None]:
        if candidate_id is None:
            continue

        if candidate_id in failed_generators:
            continue

        cached = generators.get(candidate_id)
        if cached is not None:
            if was_alias:
                generators[model_id] = cached
            return cached

        try:
            tokenizer = AutoTokenizer.from_pretrained(candidate_id)
            if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(candidate_id)
            model.eval()

            if torch.cuda.is_available():
                device = 0
                model.to("cuda")
            else:
                device = -1

            gen = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device=device,
            )
        except (GatedRepoError, HTTPError, OSError) as exc:
            failed_generators[candidate_id] = exc
            continue

        generators[candidate_id] = gen
        if was_alias or candidate_id != model_id:
            generators[model_id] = gen

        if candidate_id == FALLBACK_MODEL_ID and resolved_id != FALLBACK_MODEL_ID:
            print(
                "[llm] Falling back to", FALLBACK_MODEL_ID,
                "(default model unavailable or gated)."
            )
        return gen

    failure = failed_generators.get(resolved_id) or failed_generators.get(model_id)
    if failure is not None:
        raise RuntimeError(
            "Unable to load language model."
            " Please ensure you have access to the requested repository or set BENCHMARK_LLM_MODEL_ID"
            " to an open model."
        ) from failure
    raise RuntimeError(
        "Unable to load language model. Set BENCHMARK_LLM_MODEL_ID to an accessible Hugging Face repo."
    )


def _truncate_article(tokenizer, article: str, max_tokens: int) -> Tuple[str, bool]:
    """Keep the prompt within the model context window while preserving the extremities."""
    tokens = tokenizer.encode(article, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return article, False

    half = max(1, max_tokens // 2)
    prefix = tokenizer.decode(tokens[:half], skip_special_tokens=True).strip()
    suffix = tokenizer.decode(tokens[-half:], skip_special_tokens=True).strip()
    placeholder = "\n\n[...truncated for context budget...]\n\n"
    truncated = prefix + placeholder + suffix
    return truncated, True


def _compute_article_token_budget(generator: TextGenerationPipeline) -> int:
    """Derive the maximum article token budget allowed by the underlying model."""
    tokenizer = generator.tokenizer
    model = getattr(generator, "model", None)

    candidates = []
    tok_limit = getattr(tokenizer, "model_max_length", None)
    if isinstance(tok_limit, int) and 0 < tok_limit < 1_000_000:
        candidates.append(int(tok_limit))

    if model is not None:
        config = getattr(model, "config", None)
        if config is not None:
            for name in ("max_position_embeddings", "n_positions", "max_sequence_length"):
                value = getattr(config, name, None)
                if isinstance(value, int) and 0 < value < 1_000_000:
                    candidates.append(int(value))

    if not candidates:
        max_context = DEFAULT_CONTEXT_TOKENS
    else:
        max_context = max(max(candidates), DEFAULT_CONTEXT_TOKENS)

    reserved = MAX_NEW_TOKENS + PROMPT_OVERHEAD_TOKENS
    if max_context > reserved:
        limit = max_context - reserved
    else:
        # If the context window is very small, keep at least half for the article.
        limit = max(32, max_context // 2)

    if max_context > MAX_NEW_TOKENS:
        limit = min(limit, max_context - MAX_NEW_TOKENS)

    if max_context > 256:
        limit = max(limit, 128)

    limit = max(32, min(limit, max_context))
    return int(limit)


def _build_prompt(article: str, question: str, truncated: bool) -> str:
    note = "The article was truncated to fit the context window. Focus on the available parts." if truncated else "Use only the article content."
    template = textwrap.dedent(
        """
        You are a meticulous research assistant. Read the article and answer the question precisely.
        {note}

        Article:
        \"\"\"
        {article}
        \"\"\"

        Question: {question}

        Please respond using exactly two labeled lines:
        predicted: <short answer>
        thinking: <brief justification or cite 'insufficient evidence'>
        """
    ).strip()
    return template.format(note=note, article=article, question=question)


def _parse_response(text: str) -> Tuple[str, str]:
    predicted = ""
    thinking = ""
    for raw_line in text.splitlines():
        line = raw_line.strip()
        lower = line.lower()
        if lower.startswith("predicted:") and not predicted:
            predicted = line.split(":", 1)[1].strip()
        elif lower.startswith("thinking:") and not thinking:
            thinking = line.split(":", 1)[1].strip()
    if not predicted:
        predicted = text.strip()
    return predicted, thinking


def llm(article: str, label: str, model: str = MODEL_ID) -> Tuple[str, str]:
    """Answer a question about an article using a causal LLM.

    Returns a tuple (predicted, other) where other contains the reasoning segment.
    """
    generator = _load_generator(model)
    tokenizer = generator.tokenizer

    limit = _compute_article_token_budget(generator)
    context, truncated = _truncate_article(tokenizer, article, limit)
    prompt = _build_prompt(context, label, truncated)

    outputs = generator(
        prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        temperature=0.2,
        top_p=0.9,
        return_full_text=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    generated_text = outputs[0]["generated_text"] if outputs else ""
    predicted, thinking = _parse_response(generated_text)
    return predicted, thinking
