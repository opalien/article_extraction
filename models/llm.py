from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Iterable, Literal, Optional, Tuple
import math
try:  # optional progress bar
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore


column_name_to_question = {
    "model": "What is the name of the proposed model in this paper ?",
}



    

@dataclass(frozen=True)
class LLMExtractorConfig:
    model_id: str
    window: int = 2000
    stride: int = 400
    max_answer_chars: int = 200
    temperature: float = 0.0
    top_p: float = 1.0
    system_prompt: str | None = None
    question_template: str = "Extract the value for '{field}' from the text. Respond with the value only."
    aggregator: Literal["first", "longest", "concat"] = "first"


def _estimate_num_windows(text_len: int, window: int, stride: int) -> int:
    if window <= 0 or stride <= 0:
        return 1
    if text_len <= window:
        return 1
    return 1 + int(math.ceil((text_len - window) / stride))


def _iter_windows(text: str, window: int, stride: int) -> Iterable[str]:
    if window <= 0 or stride <= 0:
        yield text
        return
    n = len(text)
    i = 0
    while i < n:
        yield text[i : i + window]
        if i + window >= n:
            break
        i += stride


def _aggregate(answers: list[str], mode: str) -> str:
    if not answers:
        return ""
    if mode == "concat":
        return " ".join(a for a in answers if a)
    if mode == "longest":
        return max(answers, key=len)
    # default: first
    return answers[0]


@lru_cache(maxsize=2)
def _load_text_generation_model(model_id: str) -> Tuple["Any", "Any", str]:
    try:
        from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM  # type: ignore
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("transformers/torch required for LLM usage") from exc

    cfg = AutoConfig.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    is_enc_dec = bool(getattr(cfg, "is_encoder_decoder", False))
    if is_enc_dec:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id)

    device = "cuda" if hasattr(torch, "cuda") and torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tokenizer, model, device


def _build_inputs(tokenizer: Any, question: str, context: str, system_prompt: Optional[str]) -> Tuple["Any", int, bool]:
    is_chat = hasattr(tokenizer, "apply_chat_template")
    if is_chat:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        user_text = (
            "Answer the question using only the context. Respond with the value only.\n\n"
            f"Question: {question}\n\nContext:\n{context}\n\nAnswer:"
        )
        messages.append({"role": "user", "content": user_text})
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        prompt_len = int(getattr(input_ids, "shape", [0, 0])[1])
        return input_ids, prompt_len, True
    else:
        sys_text = (system_prompt.strip() + "\n\n") if system_prompt else ""
        prompt = (
            f"{sys_text}You are an information extraction assistant. "
            f"Answer with the value only.\n\n"
            f"Question: {question}\n\nContext:\n{context}\n\nAnswer:"
        )
        enc = tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"]
        prompt_len = int(input_ids.shape[1])
        return input_ids, prompt_len, False


def _generate(
    tokenizer: Any,
    model: Any,
    device: str,
    input_ids: "Any",
    prompt_len: int,
    *,
    temperature: float,
    top_p: float,
    max_answer_chars: int,
) -> str:
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("torch not available") from exc

    input_ids = input_ids.to(device)

    do_sample = bool(temperature and temperature > 0.0)
    # rough mapping from characters to tokens; keep bounded
    approx_tokens = max(16, min(128, int(max_answer_chars / 2) + 8))

    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": approx_tokens,
        "do_sample": do_sample,
        "temperature": float(temperature) if do_sample else None,
        "top_p": float(top_p) if do_sample else None,
        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    # remove None values to avoid warnings
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

    with torch.no_grad():
        output_ids = model.generate(input_ids=input_ids, **gen_kwargs)

    if getattr(model.config, "is_encoder_decoder", False):
        decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    else:
        new_tokens = output_ids[0][prompt_len:]
        decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # lightweight postprocessing
    text = decoded.strip()
    if "\n" in text:
        text = text.split("\n", 1)[0].strip()
    if text.startswith(":"):  # residual from "Answer:" prefix
        text = text[1:].strip()
    return text[:max_answer_chars]


def extract_fn(
    text: str,
    field: str,
    *,
    model_id: str,
    window: int = 2000,
    stride: int = 400,
    max_answer_chars: int = 200,
    temperature: float = 0.0,
    top_p: float = 1.0,
    system_prompt: str | None = None,
    question_template: str = "Extract the value for '{field}' from the text. Respond with the value only.",
    aggregator: Literal["first", "longest", "concat"] = "first",
    question_map: Optional[dict[str, str]] = None,
    call_model: Optional[Callable[[str, str, dict[str, Any]], str]] = None,
) -> str:
    """
    Windowed LLM-based extractor. Accepts configuration so callers can curry.

    - text, field: required by PaperInformation pipeline
    - model_id, window, stride, etc.: tunable parameters for currying
    - call_model: optional hook to inject the actual model call, for testing
    """
    if not text or not field:
        return ""

    qm = question_map if question_map is not None else column_name_to_question
    if field not in qm:
        return None

    prompt_kwargs = {"field": field}
    question = qm.get(field) or question_template.format(**prompt_kwargs)

    answers: list[str] = []
    total = _estimate_num_windows(len(text), window, stride)
    pbar = None
    if tqdm is not None:
        pbar = tqdm(total=total, desc=f"LLM:{field}")
    for chunk in _iter_windows(text, window, stride):
        payload = {
            "model": model_id,
            "temperature": temperature,
            "top_p": top_p,
            "system": system_prompt,
            "question": question,
            "context": chunk,
            "max_chars": max_answer_chars,
        }
        if call_model is not None:
            raw = call_model(question, chunk, payload)
        else:
            tok, net, device = _load_text_generation_model(model_id)
            input_ids, prompt_len, _ = _build_inputs(tok, question, chunk, system_prompt)
            raw = _generate(
                tok,
                net,
                device,
                input_ids,
                prompt_len,
                temperature=temperature,
                top_p=top_p,
                max_answer_chars=max_answer_chars,
            )
        answer = (raw or "").strip()[:max_answer_chars]
        if answer:
            answers.append(answer)
        if pbar is not None:
            pbar.update(1)
    if pbar is not None:
        pbar.close()
    if not answers:
        return (text or "").strip()[:max_answer_chars]
    return _aggregate(answers, aggregator)


