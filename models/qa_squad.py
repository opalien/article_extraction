from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Literal, Optional
from functools import lru_cache
import math

DEFAULT_MAX_LEN = 4096
DEFAULT_STRIDE = 1024
LMAX_TOK = 30
try:  # progress bar is optional
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore


@dataclass(frozen=True)
class QAExtractorConfig:
    model_id: str
    window: int = 2000
    stride: int = 400
    max_answer_chars: int = 200
    n_best: int = 3
    aggregator: Literal["best", "longest", "concat"] = "best"


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


def _aggregate(answers: list[tuple[str, float]], mode: str) -> str:
    if not answers:
        return ""
    if mode == "concat":
        return " ".join(a for a, _ in answers if a)
    if mode == "longest":
        return max(answers, key=lambda x: len(x[0]))[0]
    # default: best (highest score)
    return max(answers, key=lambda x: x[1])[0]


def _estimate_num_windows(text_len: int, window: int, stride: int) -> int:
    if window <= 0 or stride <= 0:
        return 1
    if text_len <= window:
        return 1
    return 1 + int(math.ceil((text_len - window) / stride))


@lru_cache(maxsize=2)
def _load_qa_model(model_id: str):
    try:
        from transformers import AutoTokenizer, AutoModelForQuestionAnswering  # type: ignore
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("transformers/torch required for QA usage") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(model_id)
    device = "cuda" if hasattr(torch, "cuda") and torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tokenizer, model, device


def _qa_candidates(
    question: str,
    context: str,
    *,
    model_id: str,
    n_best: int,
    max_answer_chars: int,
) -> list[tuple[str, float]]:
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("torch not available") from exc

    tok, net, device = _load_qa_model(model_id)

    tokenizer_max_len = getattr(tok, "model_max_length", DEFAULT_MAX_LEN)
    if isinstance(tokenizer_max_len, int) and tokenizer_max_len > 0:
        max_length = tokenizer_max_len
    else:
        max_length = DEFAULT_MAX_LEN
    stride = min(DEFAULT_STRIDE, max_length // 2) if max_length else DEFAULT_STRIDE
    if stride <= 0:
        stride = DEFAULT_STRIDE

    enc = tok(
        question,
        context,
        return_offsets_mapping=True,
        return_overflowing_tokens=True,
        truncation="only_second",
        max_length=max_length,
        stride=stride,
        padding=False,
    )
    chunks = len(enc["input_ids"])
    if chunks == 0:
        return []

    candidates: list[tuple[str, float]] = []
    for i in range(chunks):
        inputs = {
            k: torch.tensor(v[i]).unsqueeze(0).to(device)
            for k, v in enc.items()
            if k in ["input_ids", "attention_mask"]
        }
        with torch.no_grad():
            out = net(**inputs)
            start_logits = out.start_logits[0].detach().cpu()
            end_logits = out.end_logits[0].detach().cpu()

        seq_ids = enc.sequence_ids(i)
        offsets = enc["offset_mapping"][i]
        ctx_tok_idx = [t for t, s in enumerate(seq_ids) if s == 1 and offsets[t] is not None]
        if not ctx_tok_idx:
            continue

        s = start_logits[ctx_tok_idx]
        e = end_logits[ctx_tok_idx]
        N = s.numel()
        # joint score matrix with upper-triangular and max length constraint
        ii = torch.arange(N)
        joint = s[:, None] + e[None, :]
        valid = torch.triu(torch.ones_like(joint, dtype=torch.bool)) & ((ii[None, :] - ii[:, None] + 1) <= LMAX_TOK)
        if not bool(valid.any()):
            continue
        val = joint.masked_fill(~valid, float("-inf"))
        flat_idx = int(torch.argmax(val))
        idx_i, idx_j = divmod(flat_idx, N)
        tok_i = ctx_tok_idx[int(idx_i)]
        tok_j = ctx_tok_idx[int(idx_j)]
        st_char, _ = offsets[tok_i]
        _, ed_char = offsets[tok_j]
        if st_char is None or ed_char is None or ed_char <= st_char:
            continue
        ans = context[st_char:ed_char].strip()
        if not ans:
            continue
        score = float(val.view(-1)[flat_idx].item())
        candidates.append((ans[:max_answer_chars], score))

    # Deduplicate by answer text, keep best score
    best: dict[str, float] = {}
    for a, sc in candidates:
        if a not in best or sc > best[a]:
            best[a] = sc
    ranked = sorted(best.items(), key=lambda x: x[1], reverse=True)
    return [(a, sc) for a, sc in ranked[: max(1, n_best)]]


def extract_fn(
    text: str,
    field: str,
    *,
    model_id: str,
    window: int = 500,
    stride: int = 200,
    max_answer_chars: int = 200,
    n_best: int = 3,
    aggregator: Literal["best", "longest", "concat"] = "best",
    question_map: Optional[dict[str, str]] = None,
    call_qa: Optional[Callable[[str, str, dict[str, Any]], list[tuple[str, float]]]] = None,
) -> str:
    """
    Windowed QA (SQuAD-style) extractor. Accepts configuration so callers can curry.

    - text, field: required by PaperInformation pipeline
    - model_id, window, stride, etc.: tunable parameters for currying
    - call_qa: optional hook that returns list of (answer, score)
    """
    if not text or not field:
        return ""

    qm = question_map or {}
    if field not in qm:
        return None  # unmapped fields -> NULL in DB
    question = qm[field]

    answers: list[tuple[str, float]] = []
    total = _estimate_num_windows(len(text), window, stride)
    pbar = None
    if tqdm is not None:
        pbar = tqdm(total=total, desc=f"QA:{field}")

    for chunk in _iter_windows(text, window, stride):
        payload = {
            "model": model_id,
            "question": question,
            "context": chunk,
            "n_best": n_best,
            "max_chars": max_answer_chars,
        }
        if call_qa is not None:
            candidates = call_qa(question, chunk, payload)
        else:
            candidates = _qa_candidates(
                question,
                chunk,
                model_id=model_id,
                n_best=n_best,
                max_answer_chars=max_answer_chars,
            )
        if pbar is not None:
            pbar.update(1)
        for ans, score in candidates[: max(1, n_best)]:
            ans = (ans or "").strip()[:max_answer_chars]
            if ans:
                answers.append((ans, float(score)))
    if pbar is not None:
        pbar.close()
    if not answers:
        # Fallback: return a short snippet to avoid NULLs
        return (text or "").strip()[:max_answer_chars]

    # Show top-10 proposals in terminal (dedup by answer, keep best score)
    best: dict[str, float] = {}
    for a, sc in answers:
        if a not in best or sc > best[a]:
            best[a] = sc
    ranked = sorted(best.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"[qa_squad][top10] field={field}")
    for i, (a, sc) in enumerate(ranked, 1):
        snip = a.replace("\n", " ")
        if len(snip) > 120:
            snip = snip[:117] + "…"
        print(f"  {i:2d}. score={sc:.2f}  {snip}")

    return _aggregate(answers, aggregator)


