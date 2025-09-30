"""Lightweight interface to run a causal LLM for question answering over long articles."""

from __future__ import annotations

import textwrap
from typing import Dict, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers.pipelines import TextGenerationPipeline

MODEL_ID = "gemma3:4b"
DEFAULT_CONTEXT_TOKENS = 3072
MAX_NEW_TOKENS = 128
PROMPT_OVERHEAD_TOKENS = 256

generators: Dict[str, TextGenerationPipeline] = {}


def _load_generator(model_id: str):
    """Lazily load and cache a text-generation pipeline for the requested model."""
    if model_id in generators:
        return generators[model_id]

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.eval()

    if torch.cuda.is_available():
        device = 0
        model.to("cuda")
    else:
        device = -1

    generators[model_id] = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
    )
    return generators[model_id]


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
