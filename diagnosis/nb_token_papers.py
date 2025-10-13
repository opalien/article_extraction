from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_DB = "data/epoch.db"
DEFAULT_TABLE = "paper_text"

# Defaults aligned with usage elsewhere in the project
DEFAULT_BIGBIRD_ID = "FredNajjar/bigbird-QA-squad_v2.3"
DEFAULT_BERT_ID = "bert-base-uncased"
DEFAULT_LLM_ID = "gpt2"  # override via --llm-model-id as needed


@dataclass(frozen=True)
class TokenizerSpec:
    name: str
    model_id: str


def _load_texts(db_path: str, table: str, limit: Optional[int]) -> list[tuple[int, str]]:
    conn = sqlite3.connect(db_path)
    try:
        q = f"SELECT id_paper, text FROM {table}"
        if limit and limit > 0:
            q += f" LIMIT {int(limit)}"
        df = pd.read_sql_query(q, conn)
        if df.empty:
            return []
        return [(int(r["id_paper"]), str(r["text"])) for _, r in df.iterrows()]
    finally:
        conn.close()


def _count_tokens(texts: Iterable[str], model_id: str) -> list[int]:
    try:
        from transformers import AutoTokenizer  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("transformers is required to run this script") from exc

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    lengths: list[int] = []
    for txt in texts:
        enc = tok(
            txt,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            truncation=False,
        )
        input_ids = enc["input_ids"]
        lengths.append(len(input_ids))
    return lengths


def _describe(values: list[int]) -> dict[str, float]:
    if not values:
        return {"count": 0.0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=np.int64)
    return {
        "count": float(arr.size),
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


def plot_distributions(lengths_map: dict[str, list[int]], *, bins: int = 60) -> None:
    # Ensure a consistent subplot order when available
    preferred = ["bigbird", "bert", "llm"]
    names = [n for n in preferred if n in lengths_map] + [
        n for n in lengths_map.keys() if n not in preferred
    ]
    num = len(names)
    if num == 0:
        return
    fig, axes = plt.subplots(1, num, figsize=(5 * num, 4))
    if num == 1:
        axes = [axes]

    for ax, name in zip(axes, names):
        vals = [int(v) for v in lengths_map.get(name, []) if int(v) > 0]
        if not vals:
            ax.set_title(f"{name} (no data)")
            ax.set_xlabel("Token count (log)")
            ax.set_ylabel("Frequency")
            continue

        vmin = max(1, min(vals))
        vmax = max(vals)
        # Log-spaced bins to match log-scale x-axis
        edges = np.logspace(np.log10(vmin), np.log10(vmax), bins)
        ax.hist(vals, bins=edges, alpha=0.7, color="C0", edgecolor="black", linewidth=0.2)
        ax.set_xscale("log")
        ax.set_title(name)
        ax.set_xlabel("Token count (log)")
        ax.set_ylabel("Frequency")
        ax.grid(True, which="both", axis="x", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig("token_distribution.png")


def main() -> None:
    ap = argparse.ArgumentParser(description="Count tokens per paper and plot distributions.")
    ap.add_argument("--db", default=DEFAULT_DB, help="Path to SQLite DB (default: data/epoch.db)")
    ap.add_argument("--table", default=DEFAULT_TABLE, help="Table name containing texts (default: paper_text)")
    ap.add_argument("--limit", type=int, default=None, help="Optional limit of rows to process")
    ap.add_argument("--bigbird-model-id", default=DEFAULT_BIGBIRD_ID, help="HF model id for BigBird tokenizer")
    ap.add_argument("--bert-model-id", default=DEFAULT_BERT_ID, help="HF model id for BERT tokenizer")
    ap.add_argument("--llm-model-id", default=DEFAULT_LLM_ID, help="HF model id for LLM tokenizer")
    ap.add_argument("--no-plot", action="store_true", help="Skip matplotlib plot and only print stats")
    args = ap.parse_args()

    rows = _load_texts(args.db, args.table, args.limit)
    if not rows:
        print("No rows found in table.")
        return
    ids, texts = zip(*rows)

    specs = [
        TokenizerSpec("bigbird", args.bigbird_model_id),
        TokenizerSpec("bert", args.bert_model_id),
        TokenizerSpec("llm", args.llm_model_id),
    ]

    lengths_map: dict[str, list[int]] = {}
    for spec in specs:
        print(f"[tok] counting with {spec.name} ({spec.model_id}) on {len(texts)} texts…")
        lengths_map[spec.name] = _count_tokens(texts, spec.model_id)

    # Print summaries
    print()
    print("Summary stats (tokens):")
    for name, vals in lengths_map.items():
        d = _describe(vals)
        print(
            f"- {name}: count={int(d['count'])} mean={d['mean']:.1f} p50={d['p50']:.0f} p90={d['p90']:.0f} "
            f"p95={d['p95']:.0f} max={d['max']:.0f}"
        )

    if not args.no_plot:
        plot_distributions(lengths_map)


if __name__ == "__main__":
    main()


