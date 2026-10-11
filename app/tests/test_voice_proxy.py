import asyncio
import json

import pytest
import websockets
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from edge_workbench.app import create_app
from edge_workbench.config import Settings


class FakeRemote:
    def __init__(self, messages=None, failure=None, wait_for_sent=0):
        self.sent = []
        self.messages = list(messages or [])
        self.failure = failure
        self.wait_for_sent = wait_for_sent

    async def __aenter__(self):
        if self.failure:
            raise self.failure
        return self

    async def __aexit__(self, *args):
        return False

    async def send(self, message):
        self.sent.append(message)

    def __aiter__(self):
        return self

    async def __anext__(self):
        while len(self.sent) < self.wait_for_sent:
            await asyncio.sleep(0)
        if self.messages:
            return self.messages.pop(0)
        raise StopAsyncIteration


def app(settings=None):
    return create_app(settings or Settings())


def test_voice_url_is_optional_and_empty_url_closes_without_transcript():
    client = TestClient(app(Settings(voice_service_url="")))
    with client.websocket_connect("/api/voice/stream") as ws:
        message = ws.receive_json()
        assert message == {
            "type": "error",
            "code": "voice_unavailable",
            "message": "Local Voice URL is not configured",
        }
        with pytest.raises(WebSocketDisconnect):
            ws.receive_json()


def test_query_voice_url_overrides_environment_and_https_maps_to_wss(monkeypatch):
    remote = FakeRemote(
        messages=[json.dumps({"text": "hello", "is_final": True})],
        wait_for_sent=1,
    )
    captured = {}

    def connect(url, **kwargs):
        captured.update(url=url, kwargs=kwargs)
        return remote

    monkeypatch.setattr(websockets, "connect", connect)
    client = TestClient(app(Settings(voice_service_url="http://env.example:8000")))
    with client.websocket_connect(
        "/api/voice/stream?voice_url=https%3A%2F%2Fvoice.example%2Fbase%2F"
    ) as ws:
        ws.send_bytes(b"pcm")
        assert ws.receive_json() == {"text": "hello", "is_final": True}
    assert captured["url"] == "wss://voice.example/base/asr/stream?sample_rate=16000&language=auto"
    assert remote.sent == [b"pcm"]


def test_binary_pcm_and_empty_end_frame_forward_and_json_returns(monkeypatch):
    remote = FakeRemote(
        messages=[json.dumps({"text": "done", "is_final": True})],
        wait_for_sent=2,
    )
    monkeypatch.setattr(websockets, "connect", lambda url, **kwargs: remote)
    client = TestClient(app(Settings(voice_service_url="http://env.example:8000")))
    with client.websocket_connect("/api/voice/stream") as ws:
        ws.send_bytes(b"pcm-16000")
        ws.send_bytes(b"")
        assert ws.receive_json() == {"text": "done", "is_final": True}
    assert remote.sent == [b"pcm-16000", b""]


def test_upstream_failure_returns_voice_error_without_fake_text(monkeypatch):
    monkeypatch.setattr(
        websockets,
        "connect",
        lambda url, **kwargs: FakeRemote(failure=OSError("upstream unavailable")),
    )
    client = TestClient(app(Settings(voice_service_url="http://env.example:8000")))
    with client.websocket_connect("/api/voice/stream") as ws:
        message = ws.receive_json()
        assert message["type"] == "error"
        assert message["code"] == "voice_unavailable"
        assert "text" not in message
