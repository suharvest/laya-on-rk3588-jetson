from __future__ import annotations

import json
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
    kev_head_path: str = ""
    kev_rkllm_lib_path: str = ""
    kev_helper_path: str = ""
    model_services_json: str = ""
    http_timeout_seconds: float = 240.0
    model_id: str = "laya"

    def service_profiles(self) -> dict[str, dict[str, str]]:
        if not self.model_services_json:
            return {}
        try:
            value = json.loads(self.model_services_json)
        except (TypeError, ValueError):
            return {}
        if not isinstance(value, dict):
            return {}
        profiles: dict[str, dict[str, str]] = {}
        for ident, raw in value.items():
            if not isinstance(ident, str) or not isinstance(raw, dict):
                continue
            url = raw.get("service_url", raw.get("url", ""))
            name = raw.get("name", ident)
            if isinstance(url, str) and isinstance(name, str):
                profiles[ident] = {"service_url": url.strip(), "name": name.strip() or ident}
        return profiles

    @classmethod
    def from_env(cls) -> "Settings":
        backend = os.getenv("DECISION_BACKEND", "unavailable").strip().lower()
        if backend not in {"unavailable", "rknn", "rkllm", "tensorrt", "http"}:
            backend = "unavailable"
        seq = _int("SEQ_LEN", 8192 if backend == "rkllm" else 90, 1, 8192)
        try:
            timeout = float(os.getenv("HTTP_TIMEOUT_SECONDS", "240"))
        except ValueError:
            timeout = 240.0
        timeout = timeout if 1 <= timeout <= 900 else 240.0
        return cls(
            backend=backend,
            model_id=os.getenv("MODEL_ID", "kev" if backend == "rkllm" else "laya").strip() or ("kev" if backend == "rkllm" else "laya"),
            model_path=os.getenv("MODEL_PATH", ""),
            tokenizer_path=os.getenv("TOKENIZER_PATH", ""),
            seq_len=seq,
            max_options=_int("MAX_OPTIONS", 16, 2, 16),
            decision_service_url=os.getenv("DECISION_SERVICE_URL", "").strip(),
            voice_service_url=os.getenv("VOICE_SERVICE_URL", "").strip(),
            model_metadata_path=os.getenv("MODEL_METADATA_PATH", "").strip(),
            rknn_core=os.getenv("RKNN_CORE", "auto").strip(),
            kev_head_path=os.getenv("KEV_HEAD_PATH", "").strip(),
            kev_rkllm_lib_path=os.getenv("RKLLM_LIB_PATH", "").strip(),
            kev_helper_path=os.getenv("RKLLM_HELPER_PATH", "").strip(),
            model_services_json=os.getenv("MODEL_SERVICES_JSON", "").strip(),
            http_timeout_seconds=timeout,
        )