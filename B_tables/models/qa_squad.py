MODEL_ID = "FredNajjar/bigbird-QA-squad_v2.3"
DEFAULT_MAX_LEN = 4096
DEFAULT_STRIDE = 1024

label_to_question = {
    "Model": "What is the name of the proposed model?",
}


import re, html, textwrap, math
from typing import Tuple
from functools import lru_cache
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForQuestionAnswering


column_name_to_question = {
    "model": "What is the name of the model introduced in the paper?",
    "abstract": "Summarize the key contributions of the paper in one sentence.",
    "architecture": "What architecture does the paper describe for the model?",
    "parameters": "How many parameters does the model have?",
    "id_country": "Which country is primarily associated with the team behind the model?",
    "id_hardware": "Which hardware platform is used to train the model?",
    "h_compute": "How much compute (in FLOPs) is reported for the training hardware?",
    "h_power": "What is the power consumption (in watts) of the training hardware?",
    "h_number": "How many hardware units are used during training?",
    "training_time_id_hardware": "Which hardware platform is used during the main training run?",
    "training_compute": "What is the total training compute in FLOPs?",
    "power_draw": "What is the average power draw in watts during training?",
    "co2eq": "What is the reported CO2-equivalent emissions from training?",
    "year": "In which year was the model released or published?",
    "training_time": "How long does the training take in total?",
}

def extract_fn(file_content: str, column_name: str) -> str:
    question = column_name_to_question[column_name]
    answer, _ = qa_squad(file_content, question)
    if isinstance(answer, tuple):
        answer = answer[0]
    if isinstance(answer, str):
        return answer.strip()
    return ""



@lru_cache(maxsize=2)
def _load_qa_model(model_id: str = MODEL_ID) -> Tuple[AutoTokenizer, AutoModelForQuestionAnswering]:
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    qa_model = AutoModelForQuestionAnswering.from_pretrained(model_id)
    return tokenizer, qa_model


def qa_squad(article: str, label: str, model: str = MODEL_ID) -> str:
    def split_words(text: str):
        return list(re.finditer(r"\S+", text, flags=re.M))

    def rgba(alpha: float) -> str:
        alpha = max(0.0, min(1.0, float(alpha)))
        return f"background-color: rgba(255,165,0,{alpha});"

    def render_html(text, char_scores, top_spans, title="QA heatmap (inside prob)", meta_note=""):
        parts = []
        parts += ["<!doctype html><meta charset='utf-8'>",
                  f"<title>{html.escape(title)}</title>",
                  """
<style>
 body{font:16px/1.6 system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:24px;max-width:980px}
 .legend{margin-bottom:12px;color:#333}
 pre{white-space:pre-wrap;word-wrap:break-word}
 .w{padding:1px 2px;border-radius:3px}
 .topsegs h2{margin-top:28px}
 .seg{margin:8px 0;padding:8px 10px;border:1px solid #eee;border-radius:6px;background:#fafafa}
 .score{font-size:12px;color:#666}
 code{background:#f5f5f5;padding:2px 4px;border-radius:4px}
</style>
                  """]
        if meta_note:
            parts.append(f"<div class='legend'><strong>Note:</strong> {html.escape(meta_note)}</div>")
        parts.append("<div class='legend'>Heatmap = ∑<sub>i≤k≤j</sub> p(span(i,j)) projetée aux caractères. Plus c’est orange, plus c’est probable.</div>")
        parts.append("<pre>")
        pos = 0
        for m in split_words(text):
            s, e = m.span()
            if s > pos: parts.append(html.escape(text[pos:s]))
            w = float(char_scores[s:e].max().item()) if e > s else 0.0
            style = rgba(w) if w > 0 else ""
            parts.append(f"<span class='w' style='{style}'>" + html.escape(text[s:e]) + "</span>")
            pos = e
        if pos < len(text): parts.append(html.escape(text[pos:]))
        parts.append("</pre>")
        parts.append("<div class='topsegs'><h2>Top spans</h2>")
        if not top_spans:
            parts.append("<p><em>Aucun segment détecté.</em></p>")
        else:
            parts.append("<ol>")
            for s in top_spans:
                snip = re.sub(r"\s+", " ", s['text']).strip()
                if len(snip) > 400: snip = snip[:397] + "…"
                parts.append("<li class='seg'>")
                parts.append(f"<div class='score'>p≈<code>{s['prob']:.4f}</code> — score=<code>{s['score']:.2f}</code> — chars [{s['start']},{s['end']})</div>")
                parts.append(f"<div>{html.escape(snip)}</div>")
                parts.append("</li>")
            parts.append("</ol>")
        parts.append("</div>")
        return "".join(parts)

    LMAX = 30
    TOPK_SAVE = 100
    TOPK_SHOW = 10

    tok, net = _load_qa_model(model)
    net.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net.to(device)

    tokenizer_max_len = getattr(tok, "model_max_length", DEFAULT_MAX_LEN)
    if isinstance(tokenizer_max_len, int) and tokenizer_max_len > 0:
        max_length = tokenizer_max_len
    else:
        max_length = DEFAULT_MAX_LEN

    stride = min(DEFAULT_STRIDE, max_length // 2) if max_length else DEFAULT_STRIDE
    if stride <= 0:
        stride = DEFAULT_STRIDE

    enc = tok(
        label, article,
        return_offsets_mapping=True,
        return_overflowing_tokens=True,
        truncation="only_second",
        max_length=max_length,
        stride=stride,
        padding=False
    )
    n_chunks = len(enc["input_ids"])
    if n_chunks == 0:
        return ""

    char_scores = torch.zeros(len(article), dtype=torch.float32)
    win_weights = []
    partial_spans = []

    for i in range(n_chunks):
        inputs = {
            k: torch.tensor(v[i]).unsqueeze(0).to(device)
            for k, v in enc.items()
            if k in ["input_ids", "attention_mask"]
        }
        with torch.no_grad():
            out = net(**inputs)
            start_logits = out.start_logits[0].detach().cpu()
            end_logits   = out.end_logits[0].detach().cpu()

        seq_ids = enc.sequence_ids(i)
        offsets = enc["offset_mapping"][i]
        ctx_idx = [
            t
            for t, s in enumerate(seq_ids)
            if s == 1 and offsets[t] is not None and offsets[t][1] > offsets[t][0]
        ]
        if not ctx_idx:
            win_weights.append(0.0)
            continue

        s = start_logits[ctx_idx]
        e = end_logits[ctx_idx]
        N = s.numel()
        scores = s[:, None] + e[None, :]
        ii = torch.arange(N)
        lenmask = (ii[None, :] - ii[:, None] + 1) <= LMAX
        valid = torch.triu(torch.ones_like(scores, dtype=torch.bool)) & lenmask
        joint = scores.masked_fill(~valid, float("-inf"))

        cls_score = start_logits[0] + end_logits[0]
        flat = torch.cat([joint[valid], cls_score.view(1)])
        Z = torch.logsumexp(flat, dim=0)
        p_span = torch.exp(joint - Z)
        p_no = float(torch.exp(cls_score - Z))
        mass = float(1.0 - p_no)
        win_weights.append(mass)

        inside = torch.stack([p_span[:k + 1, k:].sum() for k in range(N)])

        for local_k, tok_k in enumerate(ctx_idx):
            st, ed = enc["offset_mapping"][i][tok_k]
            if ed > st:
                val = float(inside[local_k])
                prev = char_scores[st:ed]
                char_scores[st:ed] = 1.0 - (1.0 - prev) * (1.0 - val)

        if mass > 0 and valid.any():
            vals = p_span[valid]
            M = vals.numel()
            k = min(TOPK_SAVE, M)
            topv, topi = torch.topk(vals, k)
            ij = torch.nonzero(valid, as_tuple=False)[topi]
            q = (topv / mass).tolist()
            for (ii_j, jj_j), qv, vprob in zip(ij.tolist(), q, topv.tolist()):
                tok_i = ctx_idx[ii_j]
                tok_j = ctx_idx[jj_j]
                st_char, _ = enc["offset_mapping"][i][tok_i]
                _, ed_char = enc["offset_mapping"][i][tok_j]
                if ed_char <= st_char:
                    continue
                score_margin = float((s[ii_j] + e[jj_j] - cls_score).item())
                partial_spans.append(
                    {
                        "w": i,
                        "start": int(st_char),
                        "end": int(ed_char),
                        "q": float(qv),
                        "score": score_margin,
                    }
                )

    W = sum(win_weights)
    if W <= 0:
        top_list = []
        slug = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")[:40] or "qa"
        out_path = f"qa_heatmap_{slug}.html"
        html_str = render_html(
            article,
            char_scores,
            top_list,
            meta_note=(
                f"model={model}, max_len={max_length}, stride={stride}, Lmax={LMAX} | "
                "(aucune réponse détectée)"
            ),
        )
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(html_str)
        return ""

    pi = [w / W for w in win_weights]

    agg = {}
    for sp in partial_spans:
        pglob = pi[sp["w"]] * sp["q"]
        key = (sp["start"], sp["end"])
        if key not in agg:
            agg[key] = {"prob": 0.0, "score": sp["score"]}
        agg[key]["prob"] += pglob
        if sp["score"] > agg[key]["score"]:
            agg[key]["score"] = sp["score"]

    spans_sorted = sorted(
        [
            {
                "start": s,
                "end": e,
                "prob": v["prob"],
                "score": v["score"],
                "text": article[s:e],
            }
            for (s, e), v in agg.items()
        ],
        key=lambda x: (x["prob"], x["score"]),
        reverse=True,
    )
    top_list = spans_sorted[:TOPK_SHOW]

    mx = float(char_scores.max().item())
    if mx > 0:
        char_scores = char_scores / mx
    slug = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")[:40] or "qa"
    out_path = f"qa_heatmap_{slug}.html"
    html_str = render_html(
        article, char_scores,
        top_list,
        title="QA inside probability heatmap",
        meta_note=f"model={model}, max_len={max_length}, stride={stride}, Lmax={LMAX}"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_str)

    best_answer = top_list[0]["text"] if top_list else ""
    return (best_answer, top_list)
