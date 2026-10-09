from __future__ import annotations

import os
from dataclasses import dataclass


def _int(name: str, default: int, low: int, high: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if low <= value <= high else default


@dataclass(frozen=True)
class Settings:
    backend: str = "unavailable"
    model_path: str = ""
    tokenizer_path: str = ""
    seq_len: int = 90
    max_options: int = 16
    decision_service_url: str = ""
    voice_service_url: str = ""
    model_metadata_path: str = ""
    rknn_core: str = "auto"

    @classmethod
    def from_env(cls) -> "Settings":
        backend = os.getenv("DECISION_BACKEND", "unavailable").strip().lower()
        if backend not in {"unavailable", "rknn", "tensorrt", "http"}:
            backend = "unavailable"
        seq = _int("SEQ_LEN", 90, 1, 8192)
        return cls(backend, os.getenv("MODEL_PATH", ""), os.getenv("TOKENIZER_PATH", ""),
                   seq, _int("MAX_OPTIONS", 16, 2, 16), os.getenv("DECISION_SERVICE_URL", "").strip(),
                   os.getenv("VOICE_SERVICE_URL", "").strip(), os.getenv("MODEL_METADATA_PATH", "").strip(),
                   os.getenv("RKNN_CORE", "auto").strip())
