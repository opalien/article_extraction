#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, re, html, textwrap
import math
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

MODEL_ID = "FredNajjar/bigbird-QA-squad_v2.3"

def split_words(text: str):
    return list(re.finditer(r"\S+", text, flags=re.M))

def rgba(alpha: float) -> str:
    alpha = max(0.0, min(1.0, float(alpha)))
    return f"background-color: rgba(255, 165, 0, {alpha});"  # orange

def render_html(text, char_scores, top_spans, title="QA inside probability heatmap", meta_note=""):
    parts = []
    parts.append("<!doctype html><meta charset='utf-8'>")
    parts.append(f"<title>{html.escape(title)}</title>")
    parts.append("""
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
    """)
    if meta_note:
        parts.append(f"<div class='legend'><strong>Note:</strong> {html.escape(meta_note)}</div>")
    parts.append("<div class='legend'>Heatmap = ∑<sub>i≤k≤j</sub> p(span(i,j)) projetée aux caractères (plus c’est orange, plus c’est probable).</div>")
    parts.append("<pre>")
    pos = 0
    for m in split_words(text):
        s, e = m.span()
        if s > pos:
            parts.append(html.escape(text[pos:s]))
        w = float(char_scores[s:e].max().item()) if e > s else 0.0
        style = rgba(w) if w > 0 else ""
        parts.append(f"<span class='w' style='{style}'>" + html.escape(text[s:e]) + "</span>")
        pos = e
    if pos < len(text):
        parts.append(html.escape(text[pos:]))
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
            parts.append(f"<div class='score'>p≈<code>{s['prob']:.4f}</code> — margin_vs_CLS=<code>{s['margin']:.2f}</code> — chars [{s['start']},{s['end']})</div>")
            parts.append(f"<div>{html.escape(snip)}</div>")
            parts.append("</li>")
        parts.append("</ol>")
    parts.append("</div>")
    return "".join(parts)

def main():
    ap = argparse.ArgumentParser(description="BigBird QA (SQuAD v2) — heatmap + top 10 spans")
    ap.add_argument("input_txt", help="Chemin du fichier .txt")
    ap.add_argument("question", help="Question (anglais ou autre)")
    ap.add_argument("output_html", help="Chemin du .html de sortie")
    ap.add_argument("--max_answer_len", type=int, default=30, help="Longueur max de la réponse (en tokens)")
    args = ap.parse_args()

    with open(args.input_txt, "r", encoding="utf-8", errors="ignore") as f:
        context = f.read()

    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_ID)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    max_len = getattr(tok, "model_max_length", 512)  # BigBird ~4096
    enc = tok(
        args.question, context,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation="only_second",
        max_length=max_len,
        padding=False
    )
    if enc["input_ids"].shape[1] >= max_len:
        print(f"[info] le contexte a été tronqué à {max_len} tokens (modèle).")

    with torch.no_grad():
        out = model(input_ids=enc["input_ids"].to(device),
                    attention_mask=enc["attention_mask"].to(device))
    start_logits = out.start_logits[0].cpu()
    end_logits   = out.end_logits[0].cpu()

    # Indices de tokens du contexte (sequence_id == 1) et offsets -> caractères
    seq_ids = enc.sequence_ids(0)  # liste python
    offsets = enc["offset_mapping"][0].tolist()
    ctx_idx = [t for t, s in enumerate(seq_ids)
               if s == 1 and offsets[t] is not None and offsets[t][1] > offsets[t][0]]
    if not ctx_idx:
        print("Aucun token de contexte encodé.")
        open(args.output_html, "w", encoding="utf-8").write("<p>Aucun contexte encodé.</p>")
        return

    s = start_logits[ctx_idx]  # (N,)
    e = end_logits[ctx_idx]    # (N,)
    N = s.numel()
    LMAX = max(1, args.max_answer_len)

    # Conjointe des spans (i<=j, longueur<=LMAX)
    scores = s[:, None] + e[None, :]              # (N,N)
    ii = torch.arange(N)
    lenmask = (ii[None, :] - ii[:, None] + 1) <= LMAX
    valid = torch.triu(torch.ones_like(scores, dtype=torch.bool)) & lenmask
    joint = scores.masked_fill(~valid, float("-inf"))

    # Normalisation globale avec no-answer (CLS)
    cls_score = start_logits[0] + end_logits[0]   # CLS à l'indice 0
    flat = torch.cat([joint[valid], cls_score.view(1)])
    Z = torch.logsumexp(flat, dim=0)

    p_span = torch.exp(joint - Z)                 # somme = 1 - p_no
    p_no = float(torch.exp(cls_score - Z))

    # Marginale inside(k) = somme des spans qui couvrent k
    inside = torch.stack([p_span[:k+1, k:].sum() for k in range(N)])  # (N,)

    # Projection caractères pour heatmap
    char_scores = torch.zeros(len(context), dtype=torch.float32)
    for local_k, tok_k in enumerate(ctx_idx):
        st, ed = offsets[tok_k]
        if ed > st:
            val = float(inside[local_k])
            prev = char_scores[st:ed]
            # on prend le max (simple et lisible)
            char_scores[st:ed] = torch.maximum(prev, torch.tensor(val))
    mx = float(char_scores.max().item())
    if mx > 0:
        char_scores /= mx

    # Extraire top-10 spans
    vals = p_span[valid]
    M = int(vals.numel())
    k = min(10, M)
    topv, topi = torch.topk(vals, k)
    ij = torch.nonzero(valid, as_tuple=False)[topi]

    top_spans = []
    for (i_idx, j_idx), prob in zip(ij.tolist(), topv.tolist()):
        tok_i = ctx_idx[i_idx]; tok_j = ctx_idx[j_idx]
        st_char, _ = offsets[tok_i]
        _, ed_char = offsets[tok_j]
        if ed_char <= st_char:
            continue
        margin = float((s[i_idx] + e[j_idx] - cls_score).item())  # info annexe
        top_spans.append({
            "start": int(st_char),
            "end": int(ed_char),
            "prob": float(prob),
            "margin": margin,
            "text": context[st_char:ed_char]
        })

    # Tri final par prob décroissante (déjà le cas via topk, mais au cas où)
    top_spans.sort(key=lambda x: x["prob"], reverse=True)

    # Affichage terminal
    print(f"p(no-answer) ≈ {p_no:.4f}")
    print("Top segments:")
    if not top_spans:
        print("(vide)")
    else:
        for r, seg in enumerate(top_spans, 1):
            snip = re.sub(r"\s+", " ", seg["text"]).strip()
            snip = textwrap.shorten(snip, width=160, placeholder="…")
            print(f"{r:2d}. p≈{seg['prob']:.4f}  margin={seg['margin']:.2f}  [{seg['start']},{seg['end']})  {snip}")

    # HTML
    meta = f"model={MODEL_ID}, max_len={max_len}, Lmax={LMAX}; question={args.question}"
    html_str = render_html(context, char_scores, top_spans, meta_note=meta)
    with open(args.output_html, "w", encoding="utf-8") as f:
        f.write(html_str)
    print(f"HTML → {args.output_html}")

if __name__ == "__main__":
    main()
