import math

import httpx
import pytest
from fastapi.testclient import TestClient

from edge_workbench.app import create_app
from edge_workbench.config import Settings
from edge_workbench.decision import DecisionError, DecisionService, ModelInfo, NativeBackend
from edge_workbench.native import NativeError


class FakeBackend:
    ready = True
    reason = None
    info = ModelInfo("test", "fake", 90)

    async def infer(self, assembled, option_ids):
        return [float(i) for i in range(len(option_ids))]


def app_with(backend=None, tokenizer=None):
    s = Settings(backend="test", seq_len=90, max_options=16)
    return create_app(s, DecisionService(s, backend=backend or FakeBackend(), tokenizer=tokenizer))


def body(**kw):
    result = {"input_text": "turn line 3 down", "prompt": "choose one", "options": [
        {"id": "a", "label": "stop"}, {"id": "b", "label": "slow"}]}
    result.update(kw)
    return result


def test_schema_and_choice_mapping():
    r = TestClient(app_with()).post("/api/decide", json=body())
    assert r.status_code == 200
    data = r.json()
    assert data["selected_option_id"] == "b"
    assert [x["option_id"] for x in data["ranking"]] == ["b", "a"]
    assert set(data) >= {"request_id", "timings", "model", "rejected"}


def test_duplicate_ids_and_nan_rejected():
    c = TestClient(app_with())
    r = c.post("/api/decide", json=body(options=[{"id": "a", "label": "x"}, {"id": "a", "label": "y"}]))
    assert r.status_code == 422

    class Bad(FakeBackend):
        async def infer(self, assembled, option_ids):
            return [math.nan] * len(option_ids)
    r = TestClient(app_with(Bad())).post("/api/decide", json=body())
    assert r.status_code == 502 or r.status_code == 503


def test_input_too_long_is_not_truncated():
    s = Settings(backend="test", seq_len=10, max_options=16)
    c = TestClient(create_app(s, DecisionService(s, backend=FakeBackend())))
    r = c.post("/api/decide", json=body(input_text="x" * 100))
    assert r.status_code == 422
    assert r.json()["detail"]["code"] == "input_too_long"
    assert r.json()["detail"]["max_tokens"] == 10 if "max_tokens" in r.json()["detail"] else True


def test_unavailable_is_503_and_not_fake():
    s = Settings()
    c = TestClient(create_app(s))
    assert c.get("/api/health").json()["model_ready"] is False
    r = c.post("/api/decide", json=body())
    assert r.status_code == 503
    assert r.json()["detail"]["code"] == "model_unavailable"


def test_unavailable_default_can_use_remote_override(monkeypatch):
    captured = {}
    class Response:
        status_code = 200
        def json(self):
            return {"selected_option_id": "b", "ranking": [{"option_id": "b", "score": 2}, {"option_id": "a", "score": 1}], "rejected": False,
                    "timings": {"encoding_ms": 1, "inference_ms": 2, "total_ms": 3}}
    class Client:
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def post(self, url, json): captured.update(json); return Response()
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: Client())
    c = TestClient(create_app(Settings()))
    r = c.post("/api/decide", json=body(input_text="x" * 10000, model_service_url="http://device:9000"))
    assert r.status_code == 200
    assert "model_service_url" not in captured


def test_default_unavailable_long_input_is_503():
    c = TestClient(create_app(Settings()))
    r = c.post("/api/decide", json=body(input_text="x" * 10000))
    assert r.status_code == 503
    assert r.json()["detail"]["code"] == "model_unavailable"


def test_native_codec_error_is_api_error_with_length_detail():
    class Codec:
        def encode(self, prompt, input_text, options):
            raise NativeError("input_too_long", "assembled input is too long", 422,
                              actual_tokens=193, max_tokens=192)

    backend = object.__new__(NativeBackend)
    backend.ready = True
    backend.reason = None
    backend.info = ModelInfo("rknn", "test-native", 192)
    backend.codec = Codec()
    c = TestClient(app_with(backend))
    r = c.post("/api/decide", json=body())
    assert r.status_code == 422
    assert r.json()["detail"] == {"code": "input_too_long", "message": "assembled input is too long",
                                   "actual_tokens": 193, "max_tokens": 192}


def test_remote_422_preserves_input_length_detail(monkeypatch):
    class Response:
        status_code = 422
        def json(self):
            return {"detail": {"code": "input_too_long", "message": "too long", "actual_tokens": 513, "max_tokens": 512}}
    class Client:
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def post(self, url, json): return Response()
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: Client())
    r = TestClient(create_app(Settings())).post("/api/decide", json=body(model_service_url="http://device:9000"))
    assert r.status_code == 422
    assert r.json()["detail"]["actual_tokens"] == 513


def test_remote_missing_timings_is_invalid(monkeypatch):
    class Response:
        status_code = 200
        def json(self):
            return {"selected_option_id": "b", "ranking": [{"option_id": "b", "score": 2}, {"option_id": "a", "score": 1}]}
    class Client:
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def post(self, url, json): return Response()
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: Client())
    r = TestClient(create_app(Settings())).post("/api/decide", json=body(model_service_url="http://device:9000"))
    assert r.status_code == 502
    assert r.json()["detail"]["code"] == "remote_invalid"


def test_http_health_without_url_is_unready():
    c = TestClient(create_app(Settings(backend="http")))
    r = c.get("/api/health")
    assert r.status_code == 200
    assert r.json()["model_ready"] is False
    models = c.get("/api/models").json()
    assert models["default_model_id"] == "laya"
    assert models["models"][0]["model_ready"] is False


def test_http_default_kev_confirms_remote_identity(monkeypatch):
    class Response:
        status_code = 200
        def json(self):
            return {"selected_option_id": "b", "ranking": [
                {"option_id": "b", "score": 2}, {"option_id": "a", "score": 1}],
                "model": {"model_id": "kev", "backend": "rkllm", "name": "Kev",
                          "max_sequence_tokens": 8192, "max_options": 16},
                "timings": {"encoding_ms": 1, "inference_ms": 2, "total_ms": 3}}
    class Client:
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def post(self, url, json): return Response()
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: Client())
    settings = Settings(backend="http", model_id="kev", decision_service_url="http://device:8000")
    response = TestClient(create_app(settings)).post("/api/decide", json=body())
    assert response.status_code == 200
    assert response.json()["model"]["model_id"] == "kev"


def test_http_models_use_remote_health_for_ready_and_capabilities(monkeypatch):
    class Response:
        status_code = 200
        def json(self):
            return {"status": "ok", "model_ready": True, "backend": "rkllm", "name": "Kev",
                    "model_id": "kev", "max_sequence_tokens": 8192, "max_options": 16}
    class Client:
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def get(self, url): return Response()
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: Client())
    settings = Settings(backend="http", model_id="kev", decision_service_url="http://device:8000")
    data = TestClient(create_app(settings)).get("/api/models").json()
    row = data["models"][0]
    assert data["default_model_id"] == "kev"
    assert row["model_ready"] is True
    assert row["backend"] == "rkllm"
    assert row["max_sequence_tokens"] == 8192


def test_http_default_kev_rejects_wrong_remote_identity(monkeypatch):
    class Response:
        status_code = 200
        def json(self):
            return {"selected_option_id": "b", "ranking": [
                {"option_id": "b", "score": 2}, {"option_id": "a", "score": 1}],
                "model": {"model_id": "laya", "backend": "rknn", "name": "Laya",
                          "max_sequence_tokens": 90, "max_options": 16},
                "timings": {"encoding_ms": 1, "inference_ms": 2, "total_ms": 3}}
    class Client:
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def post(self, url, json): return Response()
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: Client())
    settings = Settings(backend="http", model_id="kev", decision_service_url="http://device:8000")
    response = TestClient(create_app(settings)).post("/api/decide", json=body())
    assert response.status_code == 502
    assert response.json()["detail"]["code"] == "remote_identity_mismatch"


def test_http_default_laya_allows_legacy_missing_identity(monkeypatch):
    class Response:
        status_code = 200
        def json(self):
            return {"selected_option_id": "b", "ranking": [
                {"option_id": "b", "score": 2}, {"option_id": "a", "score": 1}],
                "timings": {"encoding_ms": 1, "inference_ms": 2, "total_ms": 3}}
    class Client:
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def post(self, url, json): return Response()
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: Client())
    settings = Settings(backend="http", model_id="laya", decision_service_url="http://device:8000")
    response = TestClient(create_app(settings)).post("/api/decide", json=body())
    assert response.status_code == 200
    assert response.json()["model"]["model_id"] == "laya"


def test_self_service_url_has_distinct_error():
    c = TestClient(create_app(Settings()))
    r = c.post("/api/health/remote", json={"service_url": "http://testserver:80"})
    assert r.status_code == 422
    assert r.json()["detail"]["code"] == "self_service_url"


@pytest.mark.anyio
async def test_remote_proxy_removes_override_and_maps_ids(monkeypatch):
    captured = {}
    class Response:
        status_code = 200
        def json(self):
            return {"request_id": "remote", "selected_option_id": "b", "ranking": [
                {"option_id": "b", "score": 2}, {"option_id": "a", "score": 1}], "model": {"backend": "x", "name": "x", "max_sequence_tokens": 90},
                    "timings": {"encoding_ms": 1, "inference_ms": 2, "total_ms": 3}}
    class Client:
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def post(self, url, json): captured.update(url=url, json=json); return Response()
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: Client())
    s = Settings(backend="http", decision_service_url="http://device:8000")
    c = TestClient(create_app(s))
    r = c.post("/api/decide", json=body(model_service_url="http://device:9000"))
    assert r.status_code == 200
    assert captured["url"].endswith("/api/decide")
    assert "model_service_url" not in captured["json"]

def test_models_endpoint_and_unknown_model():
    s = Settings(backend="test", model_services_json='{"kev":{"url":"http://edge:9000","name":"Kev"}}')
    c = TestClient(create_app(s, DecisionService(s, backend=FakeBackend())))
    data = c.get("/api/models").json()
    assert data["default_model_id"] == "laya"
    assert [m["id"] for m in data["models"]] == ["laya", "kev"]
    assert data["models"][1]["model_ready"] is False
    r = c.post("/api/decide", json=body(model_id="missing"))
    assert r.status_code == 422
    assert r.json()["detail"]["code"] == "unknown_model"


def test_unconfigured_remote_profile_is_not_ready():
    s = Settings(backend="test", model_services_json='{"kev":{"name":"Kev"}}')
    c = TestClient(create_app(s, DecisionService(s, backend=FakeBackend())))
    row = next(m for m in c.get("/api/models").json()["models"] if m["id"] == "kev")
    assert row["model_ready"] is False


def test_app_shutdown_closes_backend():
    class ClosingBackend(FakeBackend):
        def __init__(self):
            self.closed = 0
            self.ready = True
            self.reason = None
            self.info = ModelInfo("test", "fake", 90)
        def close(self):
            self.closed += 1
    backend = ClosingBackend()
    app = app_with(backend)
    with TestClient(app):
        assert backend.closed == 0
    assert backend.closed == 1


def test_models_endpoint_rejects_remote_identity_mismatch(monkeypatch):
    class Response:
        status_code = 200
        def json(self):
            return {"status": "ok", "model_ready": True, "backend": "http", "name": "Laya",
                    "model_id": "laya", "max_sequence_tokens": 90, "max_options": 16}
    class Client:
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def get(self, url): return Response()
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: Client())
    s = Settings(backend="test", model_services_json='{"kev":{"url":"http://edge:9000","name":"Kev"}}')
    c = TestClient(create_app(s, DecisionService(s, backend=FakeBackend())))
    row = next(m for m in c.get("/api/models").json()["models"] if m["id"] == "kev")
    assert row["model_ready"] is False
    assert "model_id" in row["reason"]


def test_remote_override_rejects_explicit_identity_mismatch(monkeypatch):
    class Response:
        status_code = 200
        def json(self):
            return {"selected_option_id": "b", "ranking": [
                {"option_id": "b", "score": 2}, {"option_id": "a", "score": 1}],
                "model": {"model_id": "kev", "backend": "http", "name": "Kev",
                          "max_sequence_tokens": 8192, "max_options": 16},
                "timings": {"encoding_ms": 1, "inference_ms": 2, "total_ms": 3}}
    class Client:
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def post(self, url, json): return Response()
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: Client())
    r = TestClient(create_app(Settings())).post("/api/decide", json=body(model_service_url="http://edge:9000"))
    assert r.status_code == 502
    assert r.json()["detail"]["code"] == "remote_identity_mismatch"


def test_create_app_fallback_models_catalog(monkeypatch):
    def fail_init(self, *args, **kwargs):
        raise DecisionError("native_unavailable", "forced initialization failure")
    monkeypatch.setattr(DecisionService, "__init__", fail_init)
    settings = Settings(backend="rkllm", model_services_json='{"remote":{"url":"http://edge:9000","name":"Remote"}}')
    client = TestClient(create_app(settings))
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert data["default_model_id"] == "laya"
    assert data["models"][0]["model_ready"] is False
    assert [row["id"] for row in data["models"]] == ["laya", "remote"]
