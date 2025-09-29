#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, re, html, textwrap, math
from collections import defaultdict
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

MODEL_ID = "FredNajjar/bigbird-QA-squad_v2.3"

# ---------- petits helpers UI ----------
def split_words(text: str):
    return list(re.finditer(r"\S+", text, flags=re.M))

def rgba(alpha: float) -> str:
    alpha = max(0.0, min(1.0, float(alpha)))
    return f"background-color: rgba(255,165,0,{alpha});"

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
            parts.append(f"<div class='score'>p≈<code>{s['prob']:.4f}</code> — margin_vs_CLS=<code>{s['margin']:.2f}</code> — chars [{s['start']},{s['end']})</div>")
            parts.append(f"<div>{html.escape(snip)}</div>")
            parts.append("</li>")
        parts.append("</ol>")
    parts.append("</div>")
    return "".join(parts)

def logaddexp(a, b):
    m = max(a, b)
    return m + math.log1p(math.exp(a - m) + math.exp(b - m) - 1.0) if m != -float("inf") else b

# ---------- script principal ----------
def main():
    ap = argparse.ArgumentParser(description="BigBird QA (SQuAD v2) — sliding window heatmap + top 10 spans")
    ap.add_argument("input_txt", help="Chemin du fichier .txt")
    ap.add_argument("question", help="Question")
    ap.add_argument("output_html", help="Chemin du .html de sortie")
    ap.add_argument("--stride", type=int, default=1024, help="Chevauchement (tokens) entre fenêtres")
    ap.add_argument("--max_answer_len", type=int, default=30, help="Longueur max d’un span (tokens du contexte)")
    ap.add_argument("--topk_per_chunk", type=int, default=800, help="Nb de spans conservés par chunk pour le reranking global")
    args = ap.parse_args()

    with open(args.input_txt, "r", encoding="utf-8", errors="ignore") as f:
        article = f.read()

    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_ID)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # --- stats taille article (exigé) ---
    art_tok = tok(article, add_special_tokens=False)
    article_tokens = len(art_tok["input_ids"])
    max_len = getattr(tok, "model_max_length", 4096)
    print(f"[info] article token length (sans specials) = {article_tokens}")
    print(f"[info] model_max_length = {max_len} | stride = {args.stride}")

    # --- fenêtrage automatique HF (glissant sur le contexte) ---
    enc = tok(
        args.question, article,
        return_tensors=None,                    # on manipule chunk par chunk
        return_offsets_mapping=True,
        return_overflowing_tokens=True,
        truncation="only_second",
        max_length=max_len,
        stride=args.stride,
        padding=False,
    )
    n_chunks = len(enc["input_ids"])
    print(f"[info] #chunks = {n_chunks}")

    # agrégats heatmap + global normalisation (approx avec top-k/chunk)
    char_scores = torch.zeros(len(article), dtype=torch.float32)
    span_logsum = defaultdict(lambda: -float("inf"))  # (start,end) -> logsumexp des (s_i+e_j) multi-chunks
    all_cls_logs = []  # CLS log-scores par chunk

    kept_per_chunk = []

    for w in range(n_chunks):
        input_ids = torch.tensor(enc["input_ids"][w]).unsqueeze(0).to(device)
        attn_mask = torch.tensor(enc["attention_mask"][w]).unsqueeze(0).to(device)
        offsets    = enc["offset_mapping"][w]
        seq_ids    = enc.sequence_ids(w)

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn_mask)
        start_logits = out.start_logits[0].detach().cpu()
        end_logits   = out.end_logits[0].detach().cpu()

        # tokens de contexte gardés dans ce chunk
        ctx_idx = [t for t, s in enumerate(seq_ids)
                   if s == 1 and offsets[t] is not None and offsets[t][1] > offsets[t][0]]
        N = len(ctx_idx)
        kept_per_chunk.append(N)
        print(f"[chunk {w+1:>3}/{n_chunks}] context tokens kept = {N}")

        if N == 0:
            all_cls_logs.append(float((start_logits[0] + end_logits[0]).item()))
            continue

        s = start_logits[ctx_idx]  # (N,)
        e = end_logits[ctx_idx]    # (N,)
        LMAX = max(1, args.max_answer_len)

        # matrice conjointe des spans valides
        scores = s[:, None] + e[None, :]
        ii = torch.arange(N)
        lenmask = (ii[None, :] - ii[:, None] + 1) <= LMAX
        valid = torch.triu(torch.ones_like(scores, dtype=torch.bool)) & lenmask
        joint = scores.masked_fill(~valid, float("-inf"))

        # normalisation locale (avec no-answer) -- pour HEATMAP du chunk
        cls_score = start_logits[0] + end_logits[0]
        all_cls_logs.append(float(cls_score.item()))
        flat = torch.cat([joint[valid], cls_score.view(1)])
        Z = torch.logsumexp(flat, dim=0)
        p_span = torch.exp(joint - Z)  # somme = 1 - p_no

        # inside marginale du chunk
        inside = torch.stack([p_span[:k+1, k:].sum() for k in range(N)])  # (N,)

        # projection caractères & union probabiliste (évite pics doublons)
        for local_k, tok_k in enumerate(ctx_idx):
            st, ed = offsets[tok_k]
            if ed > st:
                val = float(inside[local_k])
                prev = char_scores[st:ed]
                char_scores[st:ed] = 1.0 - (1.0 - prev) * (1.0 - val)

        # --- collecte TOP-K spans (log-scores bruts) pour normalisation GLOBALE ---
        vals = p_span[valid]  # on s'en sert pour identifier les spans forts
        M = int(vals.numel())
        ksave = min(args.topk_per_chunk, M)
        if ksave > 0:
            topv, topi = torch.topk(vals, ksave)
            ij = torch.nonzero(valid, as_tuple=False)[topi]
            for (i_idx, j_idx) in ij.tolist():
                tok_i = ctx_idx[i_idx]; tok_j = ctx_idx[j_idx]
                st_char, _ = offsets[tok_i]
                _, ed_char = offsets[tok_j]
                if ed_char <= st_char:
                    continue
                # log-score brut (sans normalisation locale)
                lg = float((s[i_idx] + e[j_idx]).item())
                key = (int(st_char), int(ed_char))
                # log-sum-exp incrémental
                cur = span_logsum[key]
                if cur == -float("inf"):
                    span_logsum[key] = lg
                else:
                    m = max(cur, lg)
                    span_logsum[key] = m + math.log(math.exp(cur - m) + math.exp(lg - m))

    # stats chunks
    if kept_per_chunk:
        avg_kept = sum(kept_per_chunk) / len(kept_per_chunk)
        print(f"[info] avg context tokens kept/chunk ≈ {avg_kept:.1f}")

    # --- normalisation GLOBALE approximée (sur spans collectés + tous les CLS) ---
    if not span_logsum:
        print("Aucun span collecté (vérifiez max_answer_len/stride).")
        with open(args.output_html, "w", encoding="utf-8") as f:
            f.write("<p>Aucun segment détecté.</p>")
        return

    Z_global = None
    # logsumexp(spans U all CLS)
    span_vals = list(span_logsum.values())
    # logsumexp stable
    m = max(span_vals + all_cls_logs)
    Z_global = m + math.log(sum(math.exp(x - m) for x in span_vals + all_cls_logs))

    # prob globale par span unique
    spans = []
    for (st, ed), lg in span_logsum.items():
        p = math.exp(lg - Z_global)
        spans.append({"start": st, "end": ed, "prob": p})

    # trier + calculer la "margin_vs_CLS" indicative (on ne l'a pas globalement; on met None)
    spans.sort(key=lambda x: x["prob"], reverse=True)
    top10 = []
    for sp in spans[:10]:
        sp["text"] = article[sp["start"]:sp["end"]]
        sp["margin"] = 0.0  # placeholder (les margins par chunk ne sont pas comparables globalement)
        top10.append(sp)

    # --- impression terminal (Top-10) ---
    print("\nTop segments (global approx):")
    for r, seg in enumerate(top10, 1):
        snip = re.sub(r"\s+", " ", seg["text"]).strip()
        snip = textwrap.shorten(snip, width=160, placeholder="…")
        print(f"{r:2d}. p≈{seg['prob']:.4f}  [{seg['start']},{seg['end']})  {snip}")

    # --- HTML heatmap ---
    mx = float(char_scores.max().item())
    if mx > 0: char_scores /= mx
    meta = (f"model={MODEL_ID}, model_max_length={max_len}, stride={args.stride}, "
            f"Lmax={args.max_answer_len}, topk_per_chunk={args.topk_per_chunk}, "
            f"article_tokens={article_tokens}, chunks={n_chunks}")
    html_str = render_html(article, char_scores, top10, meta_note=meta)
    with open(args.output_html, "w", encoding="utf-8") as f:
        f.write(html_str)
    print(f"\nHTML → {args.output_html}")

if __name__ == "__main__":
    main()
