"""Lightweight interface to run a causal LLM for question answering over long articles."""

from __future__ import annotations

import os
import textwrap
from typing import Dict, Tuple

import torch
from huggingface_hub.errors import GatedRepoError
from requests import HTTPError
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import BitsAndBytesConfig
from transformers.pipelines import TextGenerationPipeline

MODEL_ID = "Qwen/Qwen3-13B-Instruct"
DEFAULT_CONTEXT_TOKENS = 3072
MAX_NEW_TOKENS = 128
PROMPT_OVERHEAD_TOKENS = 256

# Manual overrides for max context windows (prompt + generation) in tokens.
MODEL_CONTEXT_OVERRIDES = {
    "Qwen/Qwen3-13B": 100_000,
    "Qwen/Qwen3-13B-Instruct": 100_000,
#    "qwen/qwen3-8b": 100_000,
#    "qwen/qwen3-8b-instruct": 100_000,
#    "Qwen/Qwen3-32B": 100_000,
#    "Qwen/Qwen3-32B-Instruct": 100_000,
#    "qwen/qwen3-32b": 100_000,
#    "qwen/qwen3-32b-instruct": 100_000,
#    "Qwen/Qwen2-8B": 100_000,
#    "Qwen/Qwen2-8B-Instruct": 100_000,
#    "qwen/qwen2-8b": 100_000,
#    "qwen/qwen2-8b-instruct": 100_000,
#    "Qwen8-8B": 100_000,
#    "qwen8-8b": 100_000,
}

generators: Dict[str, TextGenerationPipeline] = {}
failed_generators: Dict[str, Exception] = {}


def _select_cuda_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    try:
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
    except AttributeError:
        pass
    return torch.float16


def _load_generator(model_id: str):
    """Charge ou récupère du cache le pipeline text-generation pour le modèle demandé."""
    if model_id in generators:
        return generators[model_id]

    if model_id in failed_generators:
        raise RuntimeError(
            f"Échec d’un chargement précédent du modèle {model_id}."
            " Assure-toi d’avoir les droits d’accès et une authentification Hugging Face valide."
        ) from failed_generators[model_id]

    auth_token = (
        os.getenv("HF_HUB_TOKEN")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
        or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )

    try:
        load_kwargs = {"trust_remote_code": True}
        if torch.cuda.is_available():
            load_kwargs["device_map"] = "auto"

            try:
                compute_dtype = _select_cuda_dtype()
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=compute_dtype,
                )
                load_kwargs["quantization_config"] = quant_config
            except Exception:
                load_kwargs["dtype"] = _select_cuda_dtype()

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            token=auth_token,
        )
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=auth_token,
            **load_kwargs,
        )
        model.eval()

        gen = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )
    except (GatedRepoError, HTTPError, OSError) as exc:
        print((GatedRepoError, HTTPError, OSError))
        failed_generators[model_id] = exc
        raise RuntimeError(
            f"Impossible de télécharger ou de charger {model_id}."
            " Vérifie que tu as accepté la licence du modèle et que ton jeton Hugging Face"
            " est configuré (HF_HUB_TOKEN ou huggingface-cli login)."
        ) from exc

    generators[model_id] = gen
    return gen


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

    override_keys = {
        MODEL_ID,
        getattr(tokenizer, "name_or_path", ""),
        getattr(model, "name_or_path", "") if model is not None else "",
    }
    for key in override_keys:
        if not key:
            continue
        override = MODEL_CONTEXT_OVERRIDES.get(key) or MODEL_CONTEXT_OVERRIDES.get(key.lower())
        if override:
            candidates.append(int(override))

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
        predicted: <few words answer>
        thinking: <brief justification or cite 'insufficient evidence' in few words>
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
        return_full_text=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    generated_text = outputs[0]["generated_text"] if outputs else ""
    predicted, thinking = _parse_response(generated_text)
    return predicted, thinking
