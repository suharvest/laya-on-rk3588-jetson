import json
import ctypes
import sys
import types

import numpy as np
import pytest

from edge_workbench.native import LayaCodec, NativeError, RKNNBackend, TensorRTBackend


class FakeEncoding:
    def __init__(self, ids):
        self.ids = ids


class FakeTokenizer:
    def no_truncation(self): pass
    def no_padding(self): pass
    def token_to_id(self, token):
        return {"[MASK]": 3, "[CLS]": 1, "[SEP]": 2, "[PAD]": 0}.get(token)

    def encode(self, text, add_special_tokens=False):
        return FakeEncoding([10 + i for i, _ in enumerate(text.split())])


@pytest.fixture
def codec(tmp_path, monkeypatch):
    (tmp_path / "tokenizer.json").write_text("{}")
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({
        "mask_token": "[MASK]", "cls_token": "[CLS]", "sep_token": "[SEP]", "pad_token": "[PAD]"
    }))
    monkeypatch.setitem(sys.modules, "tokenizers", types.SimpleNamespace(Tokenizer=types.SimpleNamespace(from_file=lambda _: FakeTokenizer())))
    return LayaCodec(str(tmp_path), seq_len=32, max_options=4)


def test_codec_markers_and_dtype(codec):
    out = codec.encode("choose", "state", [{"id": "a", "label": "one"}, {"id": "b", "label": "two", "description": "x"}])
    assert out.input_ids.shape == (1, 32)
    assert out.input_ids.dtype == np.int64
    assert out.input_ids[0, 13:].tolist() == [0] * 19
    assert out.attention_mask.dtype == np.int64
    assert out.marker_pos[0, :2].tolist() == [5, 7]
    assert out.marker_mask[0, :2].tolist() == [True, True]
    assert out.qtype.tolist() == [0]


def test_codec_does_not_truncate(codec):
    with pytest.raises(NativeError) as exc:
        codec.encode("choose", "word " * 40, [{"id": "a", "label": "one"}, {"id": "b", "label": "two"}])
    assert exc.value.code == "input_too_long"
    assert exc.value.extra["max_tokens"] == 32


class Attr:
    def __init__(self, name, typ, dims):
        self.name, self.type, self.dims, self.n_dims = name.encode(), typ, dims, len(dims)


class FakeRuntime:
    def __init__(self):
        self.rknn_runtime = self
        self.ins = [Attr(n, t, d) for n, t, d in [
            ("input_ids", 8, (1, 8)), ("attention_mask", 8, (1, 8)),
            ("marker_pos", 8, (1, 16)), ("marker_mask", 9, (1, 16)), ("qtype", 8, (1,))]]
        self.outs = [Attr("act_logits", 0, (1, 2)), Attr("choice_logits", 0, (1, 16))]

    def load_rknn(self, _): return 0
    def init_runtime(self, core_mask=None): return 0
    def get_in_out_num(self): return (5, 2)
    def get_tensor_attr(self, i, is_output=False): return (self.outs if is_output else self.ins)[i]
    def inference(self, inputs):
        assert inputs[0].dtype == np.int64
        assert inputs[3].dtype == np.bool_
        return [np.asarray([[np.nan, np.nan]], dtype=np.float32),
                np.asarray([[1.0, 2.0] + [-np.inf] * 14], dtype=np.float32)]


def test_rknn_fake_metadata_and_inference(monkeypatch):
    api = types.SimpleNamespace(RKNNLite=types.SimpleNamespace(
        NPU_CORE_AUTO=0, NPU_CORE_0=1, NPU_CORE_0_1=2, NPU_CORE_0_1_2=3))
    monkeypatch.setitem(sys.modules, "rknnlite", types.ModuleType("rknnlite"))
    monkeypatch.setitem(sys.modules, "rknnlite.api", api)
    runtime = FakeRuntime()
    b = RKNNBackend("model.rknn", 8, 16, runtime=runtime)
    assert b.ready is True
    class Req:
        option_ids = ["a", "b"]
        input_ids = np.zeros((1, 8), np.int64)
        attention_mask = np.ones((1, 8), np.int64)
        marker_pos = np.zeros((1, 16), np.int64)
        marker_mask = np.asarray([[True, True] + [False] * 14], np.bool_)
        qtype = np.zeros((1,), np.int64)
    assert b.infer_request(Req()) == [1.0, 2.0]


def test_rknn_choice_nan_rejected(monkeypatch):
    api = types.SimpleNamespace(RKNNLite=types.SimpleNamespace(NPU_CORE_AUTO=0))
    monkeypatch.setitem(sys.modules, "rknnlite", types.ModuleType("rknnlite"))
    monkeypatch.setitem(sys.modules, "rknnlite.api", api)
    runtime = FakeRuntime()
    runtime.inference = lambda inputs: [np.asarray([[np.nan, np.nan]]),
                                        np.asarray([[np.nan, 2.0] + [-np.inf] * 14])]
    b = RKNNBackend("model.rknn", 8, 16, runtime=runtime)
    class Req:
        option_ids = ["a", "b"]
        input_ids = np.zeros((1, 8), np.int64); attention_mask = input_ids.copy()
        marker_pos = np.zeros((1, 16), np.int64); marker_mask = np.asarray([[True, True] + [False] * 14], np.bool_)
        qtype = np.zeros((1,), np.int64)
    with pytest.raises(NativeError, match="non-finite"):
        b.infer_request(Req())


def test_tensorrt_fake_tuple_cuda_and_fixed_output(monkeypatch):
    class DT:
        FLOAT = object(); HALF = object(); INT32 = object(); INT64 = object(); BOOL = object(); INT8 = object(); UINT8 = object()
    class Cuda:
        cudaMemcpyKind = types.SimpleNamespace(cudaMemcpyHostToDevice=1, cudaMemcpyDeviceToHost=2)
        def __init__(self): self.next = 1; self.mem = {}
        def cudaSetDevice(self, _): return (0,)
        def cudaMalloc(self, n):
            p = self.next; self.next += 1; self.mem[p] = bytearray(n); return (0, p)
        def cudaMemcpy(self, dst, src, n, kind):
            if kind == 1: self.mem[dst][:n] = bytes((src % 251 for _ in range(n)))
            else: ctypes.memmove(dst, bytes(self.mem[src][:n]), n)
            return (0,)
        def cudaDeviceSynchronize(self): return (0,)
        def cudaFree(self, _): return (0,)
    cuda = Cuda()
    monkeypatch.setitem(sys.modules, "cuda", types.ModuleType("cuda"))
    monkeypatch.setitem(sys.modules, "cuda.cudart", cuda)
    class Mode: INPUT = 1; OUTPUT = 2
    class Engine:
        num_io_tensors = 7
        names = ["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype", "act_logits", "choice_logits"]
        def get_tensor_name(self, i): return self.names[i]
        def get_tensor_dtype(self, n): return DT.INT64 if n != "act_logits" and n != "choice_logits" else DT.FLOAT
        def get_tensor_shape(self, n): return (1, 8) if n in ("input_ids", "attention_mask") else ((1, 16) if n == "choice_logits" else ((1, 2) if n == "act_logits" else ((1, 16) if n in ("marker_pos", "marker_mask") else (1,))))
        def get_tensor_mode(self, n): return Mode.INPUT if n in self.names[:5] else Mode.OUTPUT
        def create_execution_context(self): return types.SimpleNamespace(set_tensor_address=lambda *_: None, execute_async_v3=lambda **_: True)
    events = []
    class Ctx:
        def set_tensor_address(self, *_): pass
        def execute_async_v3(self, **_): return True
        def release(self): events.append("context")
    Engine.create_execution_context = lambda self: Ctx()
    class Logger:
        WARNING = 1
        def __init__(self, *_): events.append("logger")
        def release(self): events.append("logger-release")
    class Runtime:
        def __init__(self, logger): events.append("runtime")
        def deserialize_cuda_engine(self, _): events.append("deserialize"); return Engine()
        def release(self): events.append("runtime-release")
    def engine_release(self): events.append("engine")
    Engine.release = engine_release
    trt = types.SimpleNamespace(DataType=DT, TensorIOMode=Mode, Logger=Logger, Runtime=Runtime)
    monkeypatch.setitem(sys.modules, "tensorrt", trt)
    import tempfile
    with tempfile.NamedTemporaryFile() as model:
        b = TensorRTBackend(model.name, 8, 16)
    assert b.trt_runtime is not None and b.logger is not None
    assert b.ready
    class Req:
        option_ids = ["a", "b"]
        input_ids = np.zeros((1, 8), np.int64); attention_mask = input_ids.copy()
        marker_pos = np.zeros((1, 16), np.int64); marker_mask = np.asarray([[True, True] + [False] * 14], np.bool_)
        qtype = np.zeros((1,), np.int64)
    assert b.infer_request(Req()) == [0.0, 0.0]
    b.release()
    assert events[-4:] == ["context", "engine", "runtime-release", "logger-release"]
