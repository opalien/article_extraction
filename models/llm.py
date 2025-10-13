from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Literal, Optional
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
    total = 1 if window <= 0 or stride <= 0 or len(text) <= window else 1 + int(math.ceil((len(text) - window) / stride))
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
            # Placeholder behaviour: naive heuristic until wired to a real LLM.
            raw = chunk[:max(1, min(len(chunk), max_answer_chars))]
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


