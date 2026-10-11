import ctypes
import sys
import types
import numpy as np
import pytest
from edge_workbench.decision import NativeBackend
from edge_workbench.config import Settings
from edge_workbench.kev import KEV_DELIMITER_IDS, KEV_DELIMITER_TOKENS, KevCodec, KevHead, KevRuntime, NativeError, _escape_markers

class _Enc:
    def __init__(self, ids): self.ids = ids
class _Tok:
    def no_truncation(self): pass
    def no_padding(self): pass
    def token_to_id(self, token): return dict(zip(KEV_DELIMITER_TOKENS, KEV_DELIMITER_IDS)).get(token)
    def encode(self, text, add_special_tokens=False): return _Enc([1000 + b for b in text.encode("utf-8")])

@pytest.fixture
def codec(tmp_path, monkeypatch):
    (tmp_path / "tokenizer.json").write_text("{}")
    monkeypatch.setitem(sys.modules, "tokenizers", types.SimpleNamespace(Tokenizer=types.SimpleNamespace(from_file=lambda _: _Tok())))
    return KevCodec(str(tmp_path), 128, 4)

def test_codec_exact_delimiters_escaping_option_end_and_decision(codec):
    req = codec.encode("state <|box_start|> repeated", "question", [
        {"id": "a", "label": "same", "description": ""},
        {"id": "b", "label": "same", "description": "zh"},
    ])
    ids = req.input_ids[0].tolist()
    assert ids[0] == 248060
    assert ids[req.option_end_indices[0]] == 248050
    assert ids[req.option_end_indices[1]] == 248050
    assert ids[req.decision_index] == 248062
    assert ids.count(248049) == 2
    assert 248060 not in ids[1:req.option_end_indices[0]]
    assert _escape_markers("<|fim_prefix|>") == "<" + chr(0xA6) + "fim_prefix" + chr(0xA6) + ">"
    assert req.option_ids == ["a", "b"]

def test_codec_overflow_exact_boundary(codec):
    args = [{"id": "a", "label": "a"}, {"id": "b", "label": "b"}]
    req = codec.encode("x", "q", args)
    codec.seq_len = len(req.input_ids[0])
    assert codec.encode("x", "q", args).input_ids.shape[1] == codec.seq_len
    codec.seq_len -= 1
    with pytest.raises(NativeError, match="assembled input"):
        codec.encode("x", "q", args)

def _head_file(tmp_path):
    path = tmp_path / "head.npz"
    rng = np.random.default_rng(7)
    np.savez(path, q_weight=rng.normal(size=(256, 1024)).astype(np.float32), q_bias=rng.normal(size=(256,)).astype(np.float32), k_weight=rng.normal(size=(256, 1024)).astype(np.float32), k_bias=rng.normal(size=(256,)).astype(np.float32), temperature=np.float32(2.3510958125672174))
    return path

def test_qk_head_softmax_and_shape_validation(tmp_path):
    head = KevHead(str(_head_file(tmp_path)))
    probs = head.score(np.ones((8, 1024), np.float32), [2, 4], 7)
    assert len(probs) == 2 and abs(sum(probs) - 1.0) < 1e-6
    bad = tmp_path / "bad.npz"
    np.savez(bad, q_weight=np.zeros((1, 256)), q_bias=np.zeros(256), k_weight=np.zeros((256, 1024)), k_bias=np.zeros(256), temperature=1.0)
    with pytest.raises(NativeError, match="shape"): KevHead(str(bad))

class _FakeFn:
    def __init__(self, fn): self.fn, self.argtypes, self.restype = fn, None, None
    def __call__(self, *args): return self.fn(*args)

class _FakeOfficialSDK:
    def __init__(self):
        self.clear_calls = 0; self.closed = 0
        self.kev_open = _FakeFn(self.open); self.kev_clear_cache = _FakeFn(self.clear); self.kev_infer = _FakeFn(self.infer); self.kev_close = _FakeFn(self.close)
    def open(self, model, context_len, out): ctypes.cast(out, ctypes.POINTER(ctypes.c_void_p))[0] = ctypes.c_void_p(0x1234); assert context_len == 512; return 0
    def clear(self, ctx): self.clear_calls += 1; return 0
    def infer(self, ctx, ids, n, out, capacity, dim):
        self.clear_calls += 1
        assert ctx.value == 0x1234 and capacity == n and dim == 1024
        vals = (ctypes.c_float * (n * 1024))(*([1.0] * (n * 1024))); ctypes.memmove(out, vals, ctypes.sizeof(vals)); return 0
    def close(self, ctx): self.closed += 1

def test_runtime_helper_lifecycle_and_full_hidden(monkeypatch):
    fake = _FakeOfficialSDK(); monkeypatch.setattr(ctypes, "CDLL", lambda _: fake)
    runtime = KevRuntime("model.rkllm", "unused", "helper.so", 512)
    assert runtime.ready
    req = types.SimpleNamespace(input_ids=np.arange(4, dtype=np.int32).reshape(1, 4))
    hidden = runtime.infer(req)
    assert hidden.shape == (4, 1024) and np.isfinite(hidden).all() and fake.clear_calls == 1
    runtime.release(); runtime.release(); assert fake.closed == 1

def test_runtime_missing_helper_is_unready():
    runtime = KevRuntime("model.rkllm", "", "", 512)
    assert not runtime.ready and "helper path" in (runtime.reason or "")

def test_pinned_external_tokenizer_delimiters():
    path = "/home/harvest/tmp/jev-20261010/kev/base/tokenizer.json"
    if not __import__("os").path.isfile(path):
        pytest.skip("external pinned Kev tokenizer is not installed")
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(path)
    assert [tok.token_to_id(x) for x in KEV_DELIMITER_TOKENS] == list(KEV_DELIMITER_IDS)


@pytest.mark.asyncio
async def test_native_kev_encode_order_is_input_then_prompt():
    calls = []
    class Codec:
        def encode(self, input_text, prompt, options):
            calls.append((input_text, prompt))
            return types.SimpleNamespace(input_ids=np.zeros((1, 1), dtype=np.int32), option_end_indices=[0], decision_index=0)
    class Adapter:
        ready = True
        reason = None
        def infer(self, request):
            return np.zeros((1, 1024), dtype=np.float32)
    backend = object.__new__(NativeBackend)
    backend.model_id = "kev"
    backend.info = types.SimpleNamespace(model_id="kev")
    backend.codec = Codec()
    backend.adapter = Adapter()
    backend.head = types.SimpleNamespace(score=lambda hidden, ends, decision: [1.0])
    result = await backend.infer_request({"input_text": "STATE", "prompt": "QUESTION", "options": [{"id": "x", "label": "X"}]})
    assert calls == [("STATE", "QUESTION")]


def test_external_official_encoder_frozen_goldens():
    import json, os
    tok_path = "/home/harvest/tmp/jev-20261010/kev/runtime/checkpoint/tokenizer.json"
    cases_path = "/home/harvest/tmp/jev-20261010/bekko/repo-excluded/jev-shared-cases-20261010.jsonl"
    if not (os.path.isfile(tok_path) and os.path.isfile(cases_path)):
        pytest.skip("pinned external official encoder inputs are not installed")
    goldens = json.loads((__import__("pathlib").Path(__file__).with_name("kev_goldens.json")).read_text())
    cases = {json.loads(line)["id"]: json.loads(line) for line in open(cases_path, encoding="utf-8")}
    codec = KevCodec(tok_path, 8192, 16)
    for golden in goldens:
        case = cases[golden["id"]]
        req = codec.encode(case["input"], case["prompt"], [
            {"id": o["id"], "label": o["title"], "description": o.get("description", "")}
            for o in case["options"]
        ])
        assert req.input_ids.shape[1] == golden["tokens"]
        assert req.option_end_indices == golden["option_end_indices"]
        assert req.decision_index == golden["decision_index"]


def test_native_backend_releases_runtime_when_head_load_fails(monkeypatch):
    class Runtime:
        ready = True
        reason = None
        def __init__(self, *args):
            self.released = 0
        def release(self):
            self.released += 1
    runtime = Runtime()
    monkeypatch.setattr("edge_workbench.decision.KevCodec", lambda *args: object())
    monkeypatch.setattr("edge_workbench.decision.KevRuntime", lambda *args: runtime)
    def bad_head(*args):
        raise NativeError("head_unavailable", "bad head")
    monkeypatch.setattr("edge_workbench.decision.KevHead", bad_head)
    with pytest.raises(NativeError):
        NativeBackend(Settings(backend="rkllm", model_id="kev"))
    assert runtime.released == 1
