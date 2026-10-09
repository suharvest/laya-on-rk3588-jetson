from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from .config import Settings
from .decision import DecisionError, DecisionService, _safe_service_url


class Option(BaseModel):
    id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    description: str = ""

    @field_validator("id", "label")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value


class DecideRequest(BaseModel):
    input_text: str
    prompt: str
    options: list[Option] = Field(min_length=2, max_length=16)
    model_service_url: str = ""

    @field_validator("input_text", "prompt")
    @classmethod
    def nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty")
        return value

    @field_validator("options")
    @classmethod
    def unique_ids_and_labels(cls, value: list[Option]) -> list[Option]:
        ids = [o.id for o in value]
        if len(ids) != len(set(ids)):
            raise ValueError("option ids must be unique")
        return value


class RemoteHealthRequest(BaseModel):
    service_url: str


def _detail(exc: DecisionError) -> dict[str, Any]:
    return {"code": exc.code, "message": exc.message, **exc.extra}


def create_app(settings: Settings | None = None, service: DecisionService | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    try:
        service = service or DecisionService(settings)
    except DecisionError as exc:
        # The process remains useful for health/UI even if a local tokenizer is broken.
        service = DecisionService.__new__(DecisionService)
        service.settings = settings
        from .decision import UnavailableBackend, LocalTokenizer
        service.tokenizer = LocalTokenizer("")
        service.backend = UnavailableBackend(settings, exc.message)
        service._native_lock = __import__("threading").Lock()
    app = FastAPI(title="Edge Workbench", docs_url=None, redoc_url=None)
    app.state.settings, app.state.decision_service = settings, service

    @app.get("/api/health")
    async def health(request: Request) -> dict[str, Any]:
        result = service.health()
        if settings.backend == "http" and not settings.decision_service_url:
            result.update(model_ready=False, reason="decision service URL is not configured")
        elif settings.backend == "http" and settings.decision_service_url:
            url = _safe_service_url(settings.decision_service_url)
            if not url or _is_self_url(settings.decision_service_url, request):
                result.update(model_ready=False, reason="invalid decision service URL")
            else:
                try:
                    async with httpx.AsyncClient(timeout=3.0) as client:
                        remote = await client.get(url.rstrip("/") + "/api/health")
                    if remote.status_code >= 400:
                        result.update(model_ready=False, reason=f"remote health returned HTTP {remote.status_code}")
                    else:
                        data = remote.json()
                        if not _valid_health(data):
                            result.update(model_ready=False, reason="remote health response has an invalid schema")
                        else:
                            result.update(model_ready=data["model_ready"], backend=data["backend"],
                                          name=data["name"], max_sequence_tokens=data["max_sequence_tokens"],
                                          max_options=data["max_options"], reason=data.get("reason"))
                except Exception as exc:
                    result.update(model_ready=False, reason=f"remote health request failed: {exc}")
        return {k: v for k, v in result.items() if v is not None}

    @app.post("/api/health/remote")
    async def remote_health(req: RemoteHealthRequest, request: Request) -> dict[str, Any]:
        url = _safe_service_url(req.service_url)
        if not url:
            raise HTTPException(422, {"code": "invalid_service_url", "message": "service_url must be an http(s) URL without credentials or fragment"})
        if _is_self_url(url, request):
            raise HTTPException(422, {"code": "self_service_url", "message": "This is the current workbench. Leave the model URL blank to use the local backend, or enter a different running model service."})
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(url.rstrip("/") + "/api/health")
            if response.status_code >= 400:
                raise HTTPException(502, {"code": "remote_unavailable", "message": f"remote health returned HTTP {response.status_code}"})
            data = response.json()
        except HTTPException:
            raise
        except (httpx.HTTPError, ValueError) as exc:
            raise HTTPException(502, {"code": "remote_unavailable", "message": f"remote health request failed: {exc}"}) from exc
        if not _valid_health(data):
            raise HTTPException(502, {"code": "remote_invalid", "message": "remote health response has an invalid schema"})
        return data

    @app.post("/api/decide")
    async def decide(req: DecideRequest, request: Request) -> dict[str, Any]:
        try:
            body = req.model_dump()
            if body.get("model_service_url") and not _safe_service_url(body["model_service_url"]):
                raise DecisionError("invalid_service_url", "model_service_url must be an http(s) URL without credentials or fragment", 422)
            if body.get("model_service_url") and _is_self_url(body["model_service_url"], request):
                raise DecisionError("self_service_url", "This is the current workbench. Leave the model URL blank to use the local backend, or enter a different running model service.", 422)
            if not body.get("model_service_url") and settings.backend == "http" and settings.decision_service_url and _is_self_url(settings.decision_service_url, request):
                raise DecisionError("self_service_url", "This is the current workbench. Leave the model URL blank to use the local backend, or enter a different running model service.", 422)
            result = await service.decide(body)
            return result
        except DecisionError as exc:
            raise HTTPException(exc.status, _detail(exc)) from exc

    @app.websocket("/api/voice/stream")
    async def voice_stream(ws: WebSocket, voice_url: str = "") -> None:
        await ws.accept()
        target = voice_url or settings.voice_service_url
        if not _safe_service_url(target):
            await ws.send_json({"type": "error", "code": "voice_unavailable", "message": "Local Voice URL is not configured"})
            await ws.close(code=1008)
            return
        parsed = urlsplit(target)
        scheme = "wss" if parsed.scheme == "https" else "ws"
        target = f"{scheme}://{parsed.netloc}{parsed.path.rstrip('/')}/asr/stream?sample_rate=16000&language=auto"
        try:
            import websockets
            async with websockets.connect(target, open_timeout=5, close_timeout=2) as remote:
                async def client_to_remote() -> None:
                    while True:
                        message = await ws.receive()
                        if message.get("type") == "websocket.disconnect":
                            return
                        if message.get("bytes") is not None:
                            await remote.send(message["bytes"])
                        elif message.get("text") is not None:
                            await remote.send(message["text"])

                async def remote_to_client() -> None:
                    async for message in remote:
                        if isinstance(message, bytes):
                            await ws.send_bytes(message)
                        else:
                            try:
                                await ws.send_text(message)
                            except RuntimeError:
                                return

                tasks = [asyncio.create_task(client_to_remote()), asyncio.create_task(remote_to_client())]
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                for task in done:
                    if task.exception() and not isinstance(task.exception(), WebSocketDisconnect):
                        raise task.exception()
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            try:
                await ws.send_json({"type": "error", "code": "voice_unavailable", "message": str(exc)})
                await ws.close(code=1011)
            except Exception:
                pass

    web = Path(__file__).resolve().parents[1] / "web"
    if web.is_dir():
        from fastapi.staticfiles import StaticFiles
        app.mount("/", StaticFiles(directory=str(web), html=True), name="web")
    else:
        @app.get("/", include_in_schema=False)
        async def root() -> dict[str, str]:
            return {"status": "ok", "message": "web assets are not installed"}
    return app


def _is_self_url(value: str, request: Request) -> bool:
    try:
        target = urlsplit(value)
        target_port = target.port or (443 if target.scheme == "https" else 80)
    except ValueError:
        return False
    current = request.url
    host = (target.hostname or "").lower()
    current_host = (current.hostname or "").lower()
    aliases = {"localhost", "127.0.0.1", "::1"}
    same_host = host == current_host or (host in aliases and current_host in aliases)
    current_port = current.port or (443 if current.scheme == "https" else 80)
    return same_host and target_port == current_port


def _valid_health(data: Any) -> bool:
    if not isinstance(data, dict):
        return False
    required = {"status", "model_ready", "backend", "name", "max_sequence_tokens", "max_options"}
    if not required.issubset(data) or not isinstance(data["status"], str) or not isinstance(data["backend"], str) or not isinstance(data["name"], str):
        return False
    if not isinstance(data["model_ready"], bool) or isinstance(data["max_sequence_tokens"], bool) or not isinstance(data["max_sequence_tokens"], int) or data["max_sequence_tokens"] < 1:
        return False
    return not isinstance(data["max_options"], bool) and isinstance(data["max_options"], int) and 2 <= data["max_options"] <= 16
