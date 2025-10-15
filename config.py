from __future__ import annotations

MODEL_ID = "openai/gpt-oss-20b"
MAX_CONTEXT_TOKENS = 100_000
WINDOW_STRIDE_TOKENS = 0
GENERATION_KWARGS = {"temperature": 0.0, "top_p": 1.0, "max_new_tokens": 128}
HARDWARE_MATCH_THRESHOLD = 0.90
DEFAULT_PUE = 1.20
DEFAULT_MFU = 0.30

__all__ = [
    "MODEL_ID",
    "MAX_CONTEXT_TOKENS",
    "WINDOW_STRIDE_TOKENS",
    "GENERATION_KWARGS",
    "HARDWARE_MATCH_THRESHOLD",
    "DEFAULT_PUE",
    "DEFAULT_MFU",
]
