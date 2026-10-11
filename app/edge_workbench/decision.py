from __future__ import annotations

import asyncio
import math
import os
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from .config import Settings
from .native import LayaCodec, NativeError, RKNNBackend, TensorRTBackend
from .kev import KevCodec, KevHead, KevRuntime


class DecisionError(Exception):
    def __init__(self, code: str, message: str, status: int = 503, **extra: Any):
        super().__init__(message)
        self.code, self.message, self.status, self.extra = code, message, status, extra


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...


class LocalTokenizer:
    """Local-only tokenizer adapter. It never downloads a tokenizer."""
    def __init__(self, path: str):
        self._tokenizer = None
        if path:
            if not os.path.isfile(path):
                raise DecisionError("tokenizer_unavailable", f"tokenizer file not found: {path}")
            try:
                from tokenizers import Tokenizer
                self._tokenizer = Tokenizer.from_file(path)
            except Exception as exc:
                raise DecisionError("tokenizer_unavailable", f"cannot load local tokenizer: {exc}") from exc

    def count(self, text: str) -> int:
        if self._tokenizer is not None:
            return len(self._tokenizer.encode(text).ids)
        # Remote services can validate with their own tokenizer. This conservative local
        # character count prevents silent truncation when no tokenizer is configured.
        return len(text)


@dataclass(frozen=True)
class ModelInfo:
    backend: str
    name: str
    max_sequence_tokens: int
    model_id: str = "laya"


class Backend(Protocol):
    info: ModelInfo
    ready: bool
    reason: str | None
    async def infer(self, assembled: str, option_ids: list[str]) -> list[float]: ...


class UnavailableBackend:
    def __init__(self, settings: Settings, reason: str = "no decision backend configured"):
        self.info = ModelInfo(settings.backend, os.path.basename(settings.model_path) or "unavailable",
                              settings.seq_len, settings.model_id)
        self.ready, self.reason = False, reason

    async def infer(self, assembled: str, option_ids: list[str]) -> list[float]:
        raise DecisionError("model_unavailable", self.reason or "model unavailable")


class NativeBackend(UnavailableBackend):
    """Local choice adapter. Vendor imports happen only for the selected backend."""
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model_id = "kev" if settings.backend == "rkllm" else settings.model_id
        self.info = ModelInfo(settings.backend, os.path.basename(settings.model_path) or "native",
                              settings.seq_len, self.model_id)
        self.ready, self.reason = False, None
        metadata = {}
        if settings.model_metadata_path:
            with open(settings.model_metadata_path, encoding="utf-8") as f:
                metadata = __import__("json").load(f)
        if settings.model_id == "kev" or settings.backend == "rkllm":
            self.codec = KevCodec(settings.tokenizer_path, settings.seq_len, settings.max_options)
            self.adapter = KevRuntime(settings.model_path, settings.kev_rkllm_lib_path,
                                      settings.kev_helper_path, settings.seq_len)
            try:
                self.head = KevHead(settings.kev_head_path)
            except Exception:
                self.adapter.release()
                raise
        else:
            self.codec = LayaCodec(settings.tokenizer_path, settings.seq_len, settings.max_options)
            if settings.backend == "rknn":
                self.adapter = RKNNBackend(settings.model_path, settings.seq_len, settings.max_options,
                                            settings.rknn_core, metadata=metadata)
            else:
                self.adapter = TensorRTBackend(settings.model_path, settings.seq_len, settings.max_options,
                                               metadata=metadata)
        self.ready, self.reason = self.adapter.ready, self.adapter.reason

    def close(self) -> None:
        release = getattr(getattr(self, "adapter", None), "release", None)
        if callable(release):
            release()

    async def infer_request(self, body: dict[str, Any]) -> list[float]:
        try:
            if getattr(self, "model_id", getattr(self.info, "model_id", "laya")) == "kev":
                request = self.codec.encode(body["input_text"], body["prompt"], body["options"])
                hidden = await asyncio.to_thread(self.adapter.infer, request)
                return self.head.score(hidden, request.option_end_indices, request.decision_index)
            request = self.codec.encode(body["prompt"], body["input_text"], body["options"])
            return await asyncio.to_thread(self.adapter.infer_request, request)
        except NativeError as exc:
            raise DecisionError(exc.code, exc.message, exc.status, **exc.extra) from exc


class HttpBackend:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.info = ModelInfo("http", "remote", settings.seq_len, settings.model_id)
        self.ready, self.reason = False, "remote service health has not been verified"

    async def infer(self, body: dict[str, Any], url: str) -> dict[str, Any]:
        target = _safe_service_url(url)
        if not target:
            raise DecisionError("remote_unavailable", "decision service URL is not configured")
        target = target.rstrip("/") + "/api/decide"
        payload = dict(body)
        payload.pop("model_service_url", None)
        payload.pop("model_id", None)
        try:
            async with httpx.AsyncClient(timeout=self.settings.http_timeout_seconds) as client:
                response = await client.post(target, json=payload)
        except httpx.HTTPError as exc:
            raise DecisionError("remote_unavailable", f"decision service request failed: {exc}") from exc
        if response.status_code >= 500:
            raise DecisionError("remote_unavailable", f"decision service returned HTTP {response.status_code}", 503)
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail")
            except (ValueError, AttributeError):
                detail = None
            if isinstance(detail, dict) and isinstance(detail.get("code"), str) and isinstance(detail.get("message"), str):
                extra = {k: detail[k] for k in ("actual_tokens", "max_tokens") if k in detail}
                raise DecisionError(detail["code"], detail["message"], response.status_code, **extra)
            raise DecisionError("remote_invalid", f"decision service returned HTTP {response.status_code}", 502)
        try:
            data = response.json()
        except ValueError as exc:
            raise DecisionError("remote_invalid", "decision service returned invalid JSON", 502) from exc
        return data


def _safe_service_url(value: str) -> str:
    from urllib.parse import urlsplit
    try:
        p = urlsplit(value.strip())
    except ValueError:
        return ""
    try:
        _ = p.port
    except ValueError:
        return ""
    if p.scheme not in {"http", "https"} or not p.netloc or p.username or p.password or p.fragment:
        return ""
    return value.strip()


def assemble(prompt: str, input_text: str, options: list[dict[str, str]]) -> str:
    return "\n".join([prompt, input_text] + [f"{o['id']} {o['label']} {o.get('description', '')}" for o in options])


def validate_scores(scores: Any, ids: list[str]) -> list[float]:
    if not isinstance(scores, list) or len(scores) != len(ids):
        raise DecisionError("remote_invalid", "ranking scores do not match option count", 502)
    out = []
    for score in scores:
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            raise DecisionError("remote_invalid", "ranking contains a non-finite score", 502)
        out.append(float(score))
    return out


class DecisionService:
    def __init__(self, settings: Settings, backend: Backend | None = None, tokenizer: TokenCounter | None = None):
        self.settings = settings
        if backend is not None:
            self.tokenizer = tokenizer or LocalTokenizer(settings.tokenizer_path)
            self.backend = backend
        elif settings.backend == "http":
            self.tokenizer = tokenizer or LocalTokenizer("")
            self.backend = HttpBackend(settings)
        elif settings.backend == "unavailable":
            self.tokenizer = tokenizer or LocalTokenizer("")
            self.backend = UnavailableBackend(settings)
        else:
            try:
                self.backend = NativeBackend(settings)
                self.tokenizer = tokenizer or LocalTokenizer("")
            except Exception as exc:
                self.tokenizer = tokenizer or LocalTokenizer("")
                self.backend = UnavailableBackend(settings, str(exc))
        self._native_lock = threading.Lock()
        self._profiles = settings.service_profiles()

    def close(self) -> None:
        close = getattr(self.backend, "close", None)
        if callable(close):
            close()

    def model_catalog(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        local_id = getattr(self.backend.info, "model_id", self.settings.model_id)
        rows.append({"id": local_id, "name": self.backend.info.name,
                     "service_url": "", "local": True, "is_default": True,
                     "model_ready": bool(self.backend.ready), "backend": self.backend.info.backend,
                     "max_sequence_tokens": self.settings.seq_len, "max_options": self.settings.max_options,
                     **({"reason": self.backend.reason} if not self.backend.ready and self.backend.reason else {})})
        for ident, profile in self._profiles.items():
            if ident == local_id:
                continue
            rows.append({"id": ident, "name": profile["name"], "service_url": profile["service_url"],
                         "local": False, "is_default": False, "model_ready": False,
                         "backend": "http", "max_sequence_tokens": 0, "max_options": 0,
                         "reason": "remote service health has not been verified"})
        return rows


    def health(self) -> dict[str, Any]:
        result = {"status": "ok", "model_ready": bool(self.backend.ready), "backend": self.backend.info.backend,
                  "name": self.backend.info.name, "model_id": getattr(self.backend.info, "model_id", self.settings.model_id),
                  "max_sequence_tokens": self.settings.seq_len, "max_options": self.settings.max_options}
        runtime_version = getattr(getattr(self.backend, "adapter", None), "runtime_version", None)
        if runtime_version is not None:
            result["runtime_version"] = str(runtime_version)
        if not self.backend.ready:
            result["reason"] = self.backend.reason or "model unavailable"
        return result

    async def decide(self, body: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        options = body["options"]
        ids = [o["id"] for o in options]
        selected_model_id = body.get("model_id") or getattr(self.backend.info, "model_id", self.settings.model_id)
        known_ids = {row["id"] for row in self.model_catalog()}
        if selected_model_id not in known_ids and not body.get("model_service_url"):
            raise DecisionError("unknown_model", f"unknown model_id: {selected_model_id}", 422,
                                model_id=selected_model_id)
        if selected_model_id != getattr(self.backend.info, "model_id", self.settings.model_id) and not body.get("model_service_url"):
            profile = self._profiles.get(selected_model_id)
            if profile is None:
                raise DecisionError("model_unavailable", f"model {selected_model_id} is not configured", 503,
                                    model_id=selected_model_id)
            result = await HttpBackend(self.settings).infer(body, profile["service_url"])
            return _validate_remote_result(result, ids, self.settings, started, selected_model_id)
        # A per-request remote override is authoritative, including when the local
        # default is unavailable. It must not be constrained by the local tokenizer.
        if body.get("model_service_url"):
            result = await HttpBackend(self.settings).infer(body, body["model_service_url"])
            return _validate_remote_result(result, ids, self.settings, started, selected_model_id)
        # A configured HTTP backend owns tokenization and sequence validation at the
        # target service. Do not reject a request using this process's tokenizer.
        if isinstance(self.backend, HttpBackend):
            result = await self.backend.infer(body, body.get("model_service_url") or self.settings.decision_service_url)
            return _validate_remote_result(result, ids, self.settings, started, selected_model_id)
        assembled = assemble(body["prompt"], body["input_text"], options)
        if isinstance(self.backend, NativeBackend):
            if not self.backend.ready:
                raise DecisionError("model_unavailable", self.backend.reason or "model unavailable")
            encode_ms = 0.0
            infer_started = time.perf_counter()
            try:
                scores = await self.backend.infer_request(body)
            except NativeError as exc:
                raise DecisionError(exc.code, exc.message, exc.status, **exc.extra) from exc
            infer_ms = (time.perf_counter() - infer_started) * 1000
            scores = validate_scores(scores, ids)
            order = sorted(range(len(ids)), key=lambda i: (-scores[i], i))
            return {"request_id": uuid.uuid4().hex, "selected_option_id": ids[order[0]],
                    "ranking": [{"option_id": ids[i], "score": scores[i]} for i in order],
                    "timings": {"encoding_ms": encode_ms, "inference_ms": infer_ms,
                                 "total_ms": (time.perf_counter() - started) * 1000},
                    "model": self.health() | {"model_ready": True, "model_id": getattr(self.backend.info, "model_id", self.settings.model_id)}, "rejected": False}
        if not self.backend.ready:
            raise DecisionError("model_unavailable", self.backend.reason or "model unavailable")
        actual = self.tokenizer.count(assembled)
        if actual > self.settings.seq_len:
            raise DecisionError("input_too_long", f"assembled input has {actual} tokens; maximum is {self.settings.seq_len}",
                                422, max_tokens=self.settings.seq_len, actual_tokens=actual)
        t0 = time.perf_counter()
        if self.settings.backend in {"rknn", "tensorrt"}:
            with self._native_lock:
                scores = await self.backend.infer(assembled, ids)
        else:
            scores = await self.backend.infer(assembled, ids)
        infer_ms = (time.perf_counter() - t0) * 1000
        scores = validate_scores(scores, ids)
        order = sorted(range(len(ids)), key=lambda i: (-scores[i], i))
        return {"request_id": uuid.uuid4().hex, "selected_option_id": ids[order[0]],
                "ranking": [{"option_id": ids[i], "score": scores[i]} for i in order],
                "timings": {"encoding_ms": (t0 - started) * 1000, "inference_ms": infer_ms,
                             "total_ms": (time.perf_counter() - started) * 1000},
                "model": self.health() | {"model_ready": True}, "rejected": False}


def _validate_remote_result(data: Any, ids: list[str], settings: Settings, started: float,
                         selected_model_id: str | None = None) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise DecisionError("remote_invalid", "decision service response must be an object", 502)
    remote_model = data.get("model")
    remote_id = remote_model.get("model_id") if isinstance(remote_model, dict) else None
    if remote_id is not None and remote_id != selected_model_id:
        raise DecisionError("remote_identity_mismatch",
                            f"remote service returned model_id {remote_id}, expected {selected_model_id}", 502)
    if selected_model_id and selected_model_id != settings.model_id:
        if remote_id != selected_model_id:
            raise DecisionError("remote_identity_mismatch",
                                f"remote service did not confirm model_id {selected_model_id}", 502)
    ranking = data.get("ranking")
    if not isinstance(ranking, list) or len(ranking) != len(ids):
        raise DecisionError("remote_invalid", "decision service ranking does not match options", 502)
    seen, clean = set(), []
    previous = math.inf
    for row in ranking:
        if not isinstance(row, dict) or row.get("option_id") not in ids or row["option_id"] in seen:
            raise DecisionError("remote_invalid", "decision service returned an unknown or duplicate option", 502)
        score = row.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            raise DecisionError("remote_invalid", "decision service returned a non-finite score", 502)
        score = float(score)
        if score > previous:
            raise DecisionError("remote_invalid", "decision service ranking is not ordered by score", 502)
        previous = score
        seen.add(row["option_id"]); clean.append({"option_id": row["option_id"], "score": score})
    if seen != set(ids):
        raise DecisionError("remote_invalid", "decision service omitted an option", 502)
    selected = data.get("selected_option_id")
    if selected is not None and selected not in ids:
        raise DecisionError("remote_invalid", "decision service selected an unknown option", 502)
    rejected = bool(data.get("rejected", False))
    if rejected and selected is not None:
        raise DecisionError("remote_invalid", "rejected response cannot select an option", 502)
    if not rejected and selected is None:
        raise DecisionError("remote_invalid", "non-rejected response must select an option", 502)
    timing = data.get("timings")
    if not isinstance(timing, dict) or any(key not in timing for key in ("encoding_ms", "inference_ms", "total_ms")):
        raise DecisionError("remote_invalid", "decision service timings must contain encoding_ms, inference_ms, total_ms", 502)
    timing_values = {}
    for key in ("encoding_ms", "inference_ms", "total_ms"):
        value = timing.get(key, 0)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise DecisionError("remote_invalid", f"timings.{key} is not finite", 502)
        if float(value) < 0:
            raise DecisionError("remote_invalid", f"timings.{key} must be non-negative", 502)
        timing_values[key] = float(value)
    return {"request_id": str(data.get("request_id") or uuid.uuid4().hex), "selected_option_id": selected,
            "ranking": clean, "timings": {"encoding_ms": timing_values["encoding_ms"],
            "inference_ms": timing_values["inference_ms"], "total_ms": (time.perf_counter()-started)*1000},
            "model": (data.get("model") or {"backend": "http", "name": "remote",
                      "max_sequence_tokens": settings.seq_len}) | {"model_id": selected_model_id or settings.model_id},
            "rejected": rejected, **({"reason": data["reason"]} if data.get("reason") else {})}
