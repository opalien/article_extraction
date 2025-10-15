from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator, Optional

from config import (
    GENERATION_KWARGS,
    MAX_CONTEXT_TOKENS,
    MODEL_ID,
    WINDOW_STRIDE_TOKENS,
)

_FIELD_TO_TEMPLATE = {
    "model": "questions/model.txt",
    "parameters": "questions/parameters.txt",
    "h_number": "questions/h_number.txt",
    "year": "questions/year.txt",
    "hardware_text": "questions/hardware.txt",
}

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class _ModelArtifacts:
    tokenizer: Any
    model: Any
    device: str
    is_encoder_decoder: bool
    max_context_tokens: int


@lru_cache(maxsize=None)
def _load_template_text(template_path: str) -> str:
    path = (_PROJECT_ROOT / template_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8")


@lru_cache(maxsize=2)
def _load_model_artifacts(model_id: str) -> _ModelArtifacts:
    try:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("transformers is required for LLM extraction") from exc

    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("torch is required for LLM extraction") from exc

    config = AutoConfig.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    if bool(getattr(config, "is_encoder_decoder", False)):
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        is_encoder_decoder = True
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id)
        is_encoder_decoder = False

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    max_context_tokens = _resolve_max_context_length(tokenizer, config)

    return _ModelArtifacts(
        tokenizer=tokenizer,
        model=model,
        device=device,
        is_encoder_decoder=is_encoder_decoder,
        max_context_tokens=max_context_tokens,
    )


def _resolve_max_context_length(tokenizer: Any, config: Any) -> int:
    candidates: list[int] = []
    attrs = [
        "max_position_embeddings",
        "n_positions",
        "max_sequence_length",
        "max_context_length",
        "seq_length",
    ]
    for attr in attrs:
        value = getattr(config, attr, None)
        if isinstance(value, int) and 0 < value <= 1_000_000:
            candidates.append(int(value))
    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit <= 1_000_000:
        candidates.append(int(tokenizer_limit))
    if not candidates:
        return MAX_CONTEXT_TOKENS
    detected = max(candidates)
    return min(MAX_CONTEXT_TOKENS, detected)


def _render_prompt(template_text: str, article_text: str) -> str:
    return template_text.replace("{article_text}", article_text)


def _generate_raw(
    prompt: str,
    tokenizer: Any,
    model: Any,
    device: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    is_encoder_decoder: bool,
) -> str:
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("torch is required for LLM extraction") from exc

    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    do_sample = temperature > 0.0
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature if do_sample else None,
        "top_p": top_p if do_sample else None,
        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_kwargs,
        )

    if is_encoder_decoder:
        decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    else:
        new_tokens = output_ids[0][input_ids.shape[1] :]
        decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return decoded


def _iter_article_windows(
    text: str,
    tokenizer: Any,
    window_tokens: int,
    stride_tokens: int,
) -> Iterator[list[int]]:
    if not text:
        yield []
        return

    token_ids: list[int] = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        yield []
        return

    if window_tokens <= 0 or window_tokens >= len(token_ids):
        yield token_ids
        return

    stride = stride_tokens if stride_tokens > 0 else window_tokens
    if stride <= 0:
        stride = window_tokens

    start = 0
    total = len(token_ids)
    while start < total:
        end = min(start + window_tokens, total)
        yield token_ids[start:end]
        if end >= total:
            break
        start += stride


def extract_fn(
    text: str,
    field: str,
    *,
    model_id: str = MODEL_ID,
    window_tokens: int = MAX_CONTEXT_TOKENS,
    stride_tokens: int = WINDOW_STRIDE_TOKENS,
    max_new_tokens: int = GENERATION_KWARGS["max_new_tokens"],
    temperature: float = GENERATION_KWARGS["temperature"],
    top_p: float = GENERATION_KWARGS["top_p"],
) -> Optional[str]:
    text = text or ""
    if field not in _FIELD_TO_TEMPLATE:
        return None

    template_text = _load_template_text(_FIELD_TO_TEMPLATE[field])
    artifacts = _load_model_artifacts(model_id)
    tokenizer = artifacts.tokenizer
    context_limit = artifacts.max_context_tokens

    base_prompt = _render_prompt(template_text, "")
    base_tokens = len(tokenizer.encode(base_prompt, add_special_tokens=False))
    article_budget = max(0, context_limit - base_tokens)
    if article_budget == 0:
        prompt = _render_prompt(template_text, "")
        output = _generate_raw(
            prompt,
            tokenizer,
            artifacts.model,
            artifacts.device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            is_encoder_decoder=artifacts.is_encoder_decoder,
        )
        return output if output.strip() else None

    effective_window = window_tokens if window_tokens > 0 else article_budget
    effective_window = min(effective_window, article_budget)
    stride = stride_tokens if stride_tokens > 0 else effective_window

    for token_window in _iter_article_windows(text, tokenizer, effective_window, stride):
        if not token_window:
            prompt = _render_prompt(template_text, "")
        else:
            chunk_text = tokenizer.decode(token_window, skip_special_tokens=True)
            prompt = _render_prompt(template_text, chunk_text)

        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        if len(prompt_tokens) > context_limit:
            allowed = min(article_budget, len(token_window))
            if allowed <= 0:
                continue
            truncated_text = tokenizer.decode(token_window[:allowed], skip_special_tokens=True)
            prompt = _render_prompt(template_text, truncated_text)
            prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
            if len(prompt_tokens) > context_limit:
                continue

        output = _generate_raw(
            prompt,
            tokenizer,
            artifacts.model,
            artifacts.device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            is_encoder_decoder=artifacts.is_encoder_decoder,
        )
        if output and output.strip():
            return output

    return None


__all__ = [
    "extract_fn",
    "_render_prompt",
    "_generate_raw",
    "_iter_article_windows",
]
