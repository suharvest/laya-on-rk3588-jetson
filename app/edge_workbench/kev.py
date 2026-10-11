"""Kev RKLLM token protocol, hidden bridge, and FP32 Q/K head."""
from __future__ import annotations

import ctypes
import json
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .native import NativeError

KEV_DELIMITER_IDS = (248060, 248061, 248049, 248050, 248062)
KEV_DELIMITER_TOKENS = (
    "<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>",
    "<|box_end|>", "<|fim_suffix|>",
)
_MARKER_RE = re.compile(r"<\|([^>]+)\|>")


@dataclass(frozen=True)
class KevRequest:
    input_ids: np.ndarray
    option_end_indices: list[int]
    decision_index: int
    option_ids: list[str]


def _escape_markers(value: str) -> str:
    return _MARKER_RE.sub(lambda m: "<\u00a6" + m.group(1) + "\u00a6>", value)


class KevCodec:
    def __init__(self, tokenizer_path: str, seq_len: int, max_options: int):
        p = Path(tokenizer_path)
        if p.is_dir():
            p /= "tokenizer.json"
        if not p.is_file():
            raise NativeError("tokenizer_unavailable", f"tokenizer.json not found: {p}")
        try:
            from tokenizers import Tokenizer
            self.tokenizer = Tokenizer.from_file(str(p))
            self.tokenizer.no_truncation()
            self.tokenizer.no_padding()
        except Exception as exc:
            raise NativeError("tokenizer_unavailable", f"cannot load local Kev tokenizer: {exc}") from exc
        self.seq_len, self.max_options = int(seq_len), int(max_options)
        ids = [self.tokenizer.token_to_id(token) for token in KEV_DELIMITER_TOKENS]
        if tuple(ids) != KEV_DELIMITER_IDS:
            raise NativeError("tokenizer_unavailable",
                              f"Kev delimiter IDs do not match official IDs: {ids}")

    def _encode(self, value: str) -> list[int]:
        return list(self.tokenizer.encode(_escape_markers(value), add_special_tokens=False).ids)

    def encode(self, input_text: str, prompt: str, options: list[dict[str, str]]) -> KevRequest:
        if not options:
            raise NativeError("invalid_options", "at least one option is required", 422)
        if len(options) > self.max_options:
            raise NativeError("too_many_options", f"received {len(options)} options; maximum is {self.max_options}",
                              422, max_options=self.max_options)
        ids = [KEV_DELIMITER_IDS[0], *self._encode(input_text),
               KEV_DELIMITER_IDS[1], *self._encode(prompt)]
        option_end_indices: list[int] = []
        for option in options:
            label, desc = option.get("label", ""), option.get("description", "")
            ids.extend([KEV_DELIMITER_IDS[2], *self._encode(label + (": " + desc if desc else ""))])
            option_end_indices.append(len(ids))
            ids.append(KEV_DELIMITER_IDS[3])
        decision_index = len(ids)
        ids.append(KEV_DELIMITER_IDS[4])
        if len(ids) > self.seq_len:
            raise NativeError("input_too_long", f"assembled input has {len(ids)} tokens; maximum is {self.seq_len}",
                              422, max_tokens=self.seq_len, actual_tokens=len(ids))
        return KevRequest(np.asarray([ids], dtype=np.int32), option_end_indices, decision_index,
                          [str(option["id"]) for option in options])


class KevHead:
    """CPU FP32 Q/K head from an externally supplied validated artifact."""
    def __init__(self, path: str):
        try:
            data = np.load(path, allow_pickle=False) if str(path).endswith(".npz") else None
            if data is not None:
                self.q_weight = np.asarray(data["q_weight"], np.float32)
                self.q_bias = np.asarray(data["q_bias"], np.float32)
                self.k_weight = np.asarray(data["k_weight"], np.float32)
                self.k_bias = np.asarray(data["k_bias"], np.float32)
                self.temperature = float(data["temperature"])
            else:
                obj = json.loads(Path(path).read_text())
                self.q_weight = np.asarray(obj["q_weight"], np.float32)
                self.q_bias = np.asarray(obj["q_bias"], np.float32)
                self.k_weight = np.asarray(obj["k_weight"], np.float32)
                self.k_bias = np.asarray(obj["k_bias"], np.float32)
                self.temperature = float(obj["temperature"])
        except Exception as exc:
            raise NativeError("head_unavailable", f"cannot load Kev head artifact: {exc}") from exc
        if self.q_weight.shape != (256, 1024) or self.k_weight.shape != (256, 1024):
            raise NativeError("head_unavailable", "Kev Q/K weights must each have shape (256,1024)")
        if self.q_bias.shape != (256,) or self.k_bias.shape != (256,):
            raise NativeError("head_unavailable", "Kev Q/K biases must each have shape (256,)")
        if not np.isfinite(self.q_weight).all() or not np.isfinite(self.k_weight).all() or not np.isfinite(self.q_bias).all() or not np.isfinite(self.k_bias).all() or not np.isfinite(self.temperature) or self.temperature <= 0:
            raise NativeError("head_unavailable", "Kev head contains non-finite values or invalid temperature")

    def score(self, hidden: np.ndarray, option_end_indices: list[int], decision_index: int) -> list[float]:
        h = np.asarray(hidden, dtype=np.float32)
        if h.ndim != 2 or h.shape[1] != 1024 or not np.isfinite(h).all():
            raise NativeError("native_invalid", "Kev hidden output must be finite [tokens,1024]")
        if decision_index < 0 or decision_index >= h.shape[0] or any(i < 0 or i >= h.shape[0] for i in option_end_indices):
            raise NativeError("native_invalid", "Kev head indices are outside hidden output")
        q = self.q_weight @ h[decision_index] + self.q_bias
        logits = []
        for idx in option_end_indices:
            k = self.k_weight @ h[idx] + self.k_bias
            logits.append(float(np.dot(q, k) / np.sqrt(256.0) / self.temperature))
        values = np.asarray(logits, dtype=np.float64)
        values -= np.max(values)
        probs = np.exp(values)
        probs /= np.sum(probs)
        if not np.isfinite(probs).all() or not np.isclose(float(np.sum(probs)), 1.0, atol=1e-6):
            raise NativeError("native_invalid", "Kev head softmax is non-finite")
        return [float(x) for x in probs]


class KevRuntime:
    # Runtime identity comes from external helper metadata; no ABI version claim.

    def __init__(self, model_path: str, lib_path: str, helper_path: str = "", context_len: int = 8192):
        self._lock = threading.Lock()
        self.ready = False
        self.reason: str | None = None
        self.runtime_version = None
        self._lib = None
        self._ctx = ctypes.c_void_p()
        try:
            if not helper_path:
                raise RuntimeError("RKLLM helper path is not configured")
            path = helper_path
            if lib_path:
                os.environ.setdefault("RKLLM_LIB_PATH", lib_path)
            self._lib = ctypes.CDLL(path)
            self._lib.kev_open.argtypes = [ctypes.c_char_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_void_p)]
            self._lib.kev_open.restype = ctypes.c_int
            self._lib.kev_clear_cache.argtypes = [ctypes.c_void_p]
            self._lib.kev_clear_cache.restype = ctypes.c_int
            self._lib.kev_infer.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int32,
                                            ctypes.POINTER(ctypes.c_float), ctypes.c_int32, ctypes.c_int32]
            self._lib.kev_infer.restype = ctypes.c_int
            self._lib.kev_close.argtypes = [ctypes.c_void_p]
            self._lib.kev_close.restype = None
            rc = int(self._lib.kev_open(model_path.encode(), int(context_len), ctypes.byref(self._ctx)))
            if rc != 0:
                raise RuntimeError(f"RKLLM open failed with code {rc}")
            self.ready = True
        except Exception as exc:
            self.reason = str(exc)

    def infer(self, request: KevRequest) -> np.ndarray:
        if not self.ready:
            raise NativeError("model_unavailable", self.reason or "Kev runtime unavailable")
        ids = np.ascontiguousarray(request.input_ids.reshape(-1), np.int32)
        out = np.empty((len(ids), 1024), np.float32)
        with self._lock:
            rc = int(self._lib.kev_infer(self._ctx, ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                                          len(ids), out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                          len(ids), 1024))
        if rc != 0:
            raise NativeError("native_inference_failed", f"RKLLM run failed with code {rc}")
        if not np.isfinite(out).all():
            raise NativeError("native_invalid", "RKLLM hidden output is non-finite")
        return out


    def release(self):
        with self._lock:
            if self._lib is not None and self._ctx.value:
                self._lib.kev_close(self._ctx)
                self._ctx = ctypes.c_void_p()
                self.ready = False
