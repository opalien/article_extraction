#!/usr/bin/env python3
"""Estimate the maximum context window a causal decoder model can handle on GPU.

The script performs a binary search (dichotomie) between 1 token and
``min(500_000, max_model)`` tokens, where ``max_model`` is derived from the
model configuration / tokenizer metadata. A forward pass is attempted at every
candidate sequence length and the bounds are adjusted depending on whether the
attempt succeeds (``try`` / ``except`` as requested).

Example:

    python tests/test_gpu.py --model-id google/gemma-2-4b-it --dtype float16

The script requires a CUDA-enabled environment and the requested model to be
available locally or via the Hugging Face Hub.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _infer_model_limit(model, tokenizer) -> Optional[int]:
    """Return the model's advertised maximum context length if available."""

    candidates = []

    cfg = getattr(model, "config", None)
    if cfg is not None:
        for attr in (
            "max_position_embeddings",
            "n_positions",
            "max_seq_len",
            "max_seq_length",
        ):
            value = getattr(cfg, attr, None)
            if isinstance(value, int) and value > 0:
                candidates.append(value)

    tok_max = getattr(tokenizer, "model_max_length", None)
    if tok_max and tok_max != float("inf") and tok_max < sys.maxsize:
        candidates.append(int(tok_max))

    if not candidates:
        return None

    return min(candidates)


def _resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype '{name}'. Choose from {sorted(mapping)}")
    return mapping[name]


def try_forward(model, seq_len: int, device: torch.device) -> None:
    """Attempt a forward pass with a random batch of length ``seq_len``."""

    vocab_size = int(getattr(model.config, "vocab_size", 32000))

    input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(1, seq_len),
        device=device,
        dtype=torch.long,
    )
    attention_mask = torch.ones_like(input_ids, device=device)

    with torch.inference_mode():
        _ = model(input_ids=input_ids, attention_mask=attention_mask)


def find_max_context(
    model,
    min_tokens: int,
    max_tokens: int,
    device: torch.device,
    on_attempt: Optional[Callable[[int, bool, Optional[float]], None]] = None,
) -> int:
    """Binary search (dichotomie) for the largest working sequence length."""

    best = 0
    low = min_tokens
    high = max_tokens

    iteration = 0
    while low <= high:
        iteration += 1
        mid = (low + high) // 2
        print(f"[iter {iteration}] Trying context size = {mid}")

        success = False
        torch.cuda.reset_peak_memory_stats(device)

        try:
            try_forward(model, mid, device)
        except RuntimeError as err:
            print(f"  ❌ RuntimeError at {mid} tokens: {err}")
            high = mid - 1
        except Exception as err:  # noqa: BLE001 - surface the reason
            print(f"  ❌ Failure at {mid} tokens: {err}")
            high = mid - 1
        else:
            best = mid
            low = mid + 1
            print(f"  ✅ Success, new min bound = {low}")
            success = True
        finally:
            mem_mib = None
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                mem_mib = torch.cuda.max_memory_allocated(device) / (1024**2)
                torch.cuda.empty_cache()

            if on_attempt is not None:
                on_attempt(mid, success, mem_mib)

    return best


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Binary search maximum context size on GPU")
    parser.add_argument(
        "--model-id",
        default="google/gemma-2-4b-it",
        help="Hugging Face model identifier (default: google/gemma-2-4b-it)",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Torch dtype to load the model with (default: float16)",
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=1,
        help="Minimum context size to test (default: 1)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Optional hard upper bound (will still be capped at 500k and model limit)",
    )
    parser.add_argument(
        "--plot-path",
        default="tests/gpu_vram_usage.png",
        help="Path where the VRAM usage curve (png) will be saved",
    )
    return parser.parse_args()


def main() -> None:
    if not torch.cuda.is_available():
        raise EnvironmentError("CUDA is required to run this script. No GPU detected.")

    args = parse_args()
    device = torch.device("cuda")
    dtype = _resolve_dtype(args.dtype)

    print(f"Loading tokenizer: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    print(f"Loading model: {args.model_id} (dtype={args.dtype})")
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=dtype)
    model.eval()
    model.to(device)

    model_limit = _infer_model_limit(model, tokenizer)
    upper_bound = 500_000

    if args.max_tokens:
        upper_bound = min(upper_bound, args.max_tokens)
    if model_limit:
        upper_bound = min(upper_bound, model_limit)

    upper_bound = max(args.min_tokens, upper_bound)

    print(
        "Starting binary search",
        f"min={args.min_tokens}",
        f"max={upper_bound}",
        f"model_limit={model_limit}",
        sep=" | ",
    )

    plot_path = Path(args.plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    success_tokens: list[int] = []
    success_mem: list[float] = []
    failure_tokens: list[int] = []
    failure_mem: list[float] = []

    def record_attempt(tokens: int, success: bool, mem_mib: Optional[float]) -> None:
        if success:
            if mem_mib is not None:
                success_tokens.append(tokens)
                success_mem.append(mem_mib)
        else:
            failure_tokens.append(tokens)
            failure_mem.append(mem_mib if mem_mib is not None else float("nan"))

    max_context = find_max_context(
        model=model,
        min_tokens=args.min_tokens,
        max_tokens=upper_bound,
        device=device,
        on_attempt=record_attempt,
    )

    print("-" * 80)
    print(f"Best working context size on this GPU: {max_context} tokens")

    plt.figure(figsize=(8, 5))
    if success_tokens:
        plt.plot(success_tokens, success_mem, marker="o", label="Succès", color="tab:blue")
    if failure_tokens:
        plt.scatter(failure_tokens, failure_mem, marker="x", label="Échec", color="tab:red")
    plt.xlabel("Nombre de tokens")
    plt.ylabel("Mémoire GPU max (MiB)")
    plt.title(f"Occupation VRAM pour {args.model_id}")
    plt.grid(True, linestyle="--", alpha=0.3)
    if success_tokens or failure_tokens:
        plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"Courbe VRAM sauvegardée dans {plot_path}")


if __name__ == "__main__":
    main()


