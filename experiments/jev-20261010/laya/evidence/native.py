"""Native Laya choice inference adapters.

This module deliberately has no torch/transformers dependency.  The tokenizer is read directly
from a local ``tokenizer.json`` and the device runtimes are imported only when their backend is
selected.
"""
from __future__ import annotations

import json
import math
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


class NativeError(Exception):
    def __init__(self, code: str, message: str, status: int = 503, **extra: Any):
        super().__init__(message)
        self.code, self.message, self.status, self.extra = code, message, status, extra


@dataclass(frozen=True)
class EncodedRequest:
    input_ids: np.ndarray
    attention_mask: np.ndarray
    marker_pos: np.ndarray
    marker_mask: np.ndarray
    qtype: np.ndarray
    option_ids: list[str]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as f:
            value = json.load(f)
    except (OSError, ValueError) as exc:
        raise NativeError("native_unavailable", f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise NativeError("native_unavailable", f"metadata {path} must be an object")
    return value


class LayaCodec:
    """Build the five native inputs for the choice model without Agent/torch."""

    def __init__(self, tokenizer_path: str, seq_len: int, max_options: int):
        p = Path(tokenizer_path)
        if p.is_dir():
            tokenizer_json = p / "tokenizer.json"
            config_dir = p
        else:
            tokenizer_json = p
            config_dir = p.parent
        if not tokenizer_json.is_file():
            raise NativeError("tokenizer_unavailable", f"tokenizer.json not found: {tokenizer_json}")
        try:
            from tokenizers import Tokenizer
            self.tokenizer = Tokenizer.from_file(str(tokenizer_json))
            # The native path owns all length checks; tokenizer JSON defaults must not silently
            # truncate or pad before those checks run.
            self.tokenizer.no_truncation()
            self.tokenizer.no_padding()
        except Exception as exc:
            raise NativeError("tokenizer_unavailable", f"cannot load local tokenizer: {exc}") from exc
        self.seq_len, self.max_options = int(seq_len), int(max_options)
        tc = _read_json(config_dir / "tokenizer_config.json") if (config_dir / "tokenizer_config.json").is_file() else {}
        sm = _read_json(config_dir / "special_tokens_map.json") if (config_dir / "special_tokens_map.json").is_file() else {}
        self.mask_token, self.mask_id = self._special("mask_token", tc, sm)
        self.cls_token, self.cls_id = self._special("cls_token", tc, sm)
        self.sep_token, self.sep_id = self._special("sep_token", tc, sm)
        self.pad_token, self.pad_id = self._special("pad_token", tc, sm, required=False)
        if self.pad_id is None:
            self.pad_id = 0
        if None in (self.mask_id, self.cls_id, self.sep_id):
            raise NativeError("tokenizer_unavailable", "tokenizer config must define mask/cls/sep tokens and ids")

    def _special(self, name: str, tc: dict[str, Any], sm: dict[str, Any], required: bool = True):
        value = tc.get(name, sm.get(name))
        if isinstance(value, dict):
            token = value.get("content") or value.get("token")
            ident = value.get("id")
        else:
            token, ident = value, None
        if token is None and not required:
            return None, None
        if not isinstance(token, str) or not token:
            if required:
                raise NativeError("tokenizer_unavailable", f"missing {name} in tokenizer config")
            return None, None
        if ident is None:
            ident = self.tokenizer.token_to_id(token)
        if ident is None:
            raise NativeError("tokenizer_unavailable", f"tokenizer has no id for {name} {token!r}")
        return token, int(ident)

    def _encode(self, text: str, limit: int | None = None) -> list[int]:
        # Mask text is never allowed to create an accidental active marker.
        text = text.replace(self.mask_token, " ")
        enc = self.tokenizer.encode(text, add_special_tokens=False)
        ids = list(enc.ids)
        if limit is not None and len(ids) > limit:
            raise NativeError("input_too_long", f"option exceeds {limit} tokens", 422,
                              max_tokens=limit, actual_tokens=len(ids))
        return ids

    @staticmethod
    def _state(value: str) -> str:
        # input_text is a string in the HTTP contract. Keep strict unicode when callers provide
        # structured data in tests or in a future native caller.
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False)

    def encode(self, prompt: str, input_text: str, options: list[dict[str, str]]) -> EncodedRequest:
        if len(options) > self.max_options:
            raise NativeError("too_many_options", f"received {len(options)} options; maximum is {self.max_options}", 422,
                              max_options=self.max_options)
        if not options:
            raise NativeError("invalid_options", "at least one option is required", 422)
        head = self._encode("choice question: " + prompt)
        if len(head) > 192:
            raise NativeError("input_too_long", f"head exceeds 192 tokens", 422, max_tokens=192, actual_tokens=len(head))
        option_parts: list[list[int]] = []
        for option in options:
            label = option.get("label", "")
            desc = option.get("description", "")
            text = " " + (label if not desc else f"{label}: {desc}")
            option_parts.append([self.mask_id] + self._encode(text, 48))
        head_budget_used = len(head) + sum(len(part) for part in option_parts)
        if head_budget_used > 192 or 192 - sum(len(part) for part in option_parts) < 16:
            raise NativeError("input_too_long", "choice head exceeds the 192-token budget", 422,
                              max_tokens=192, actual_tokens=head_budget_used)
        ids = [self.cls_id, *head, self.sep_id]
        markers: list[int] = []
        for part in option_parts:
            markers.append(len(ids))
            ids.extend(part)
        ids.append(self.sep_id)
        state = self._encode(self._state(input_text))
        ids.extend(state)
        ids.append(self.sep_id)
        if len(ids) > self.seq_len:
            raise NativeError("input_too_long", f"assembled input has {len(ids)} tokens; maximum is {self.seq_len}", 422,
                              max_tokens=self.seq_len, actual_tokens=len(ids))
        if len(markers) != len(options) or any(m >= len(ids) for m in markers):
            raise NativeError("input_too_long", "not all option markers fit in the sequence", 422,
                              max_tokens=self.seq_len, actual_tokens=len(ids))
        # Calibration uses zero-filled input_ids (20_build_calib.py), regardless of tokenizer
        # pad metadata.
        padded = np.zeros((1, self.seq_len), dtype=np.int64)
        padded[0, :len(ids)] = np.asarray(ids, dtype=np.int64)
        attention = np.zeros((1, self.seq_len), dtype=np.int64)
        attention[0, :len(ids)] = 1
        marker_pos = np.zeros((1, self.max_options), dtype=np.int64)
        marker_mask = np.zeros((1, self.max_options), dtype=np.bool_)
        marker_pos[0, :len(markers)] = np.asarray(markers, dtype=np.int64)
        marker_mask[0, :len(markers)] = True
        return EncodedRequest(padded, attention, marker_pos, marker_mask,
                              np.asarray([0], dtype=np.int64), [str(o["id"]) for o in options])


_RKNN_DTYPE = {0: np.float32, 1: np.float16, 2: np.int8, 3: np.uint8, 4: np.int16,
               5: np.uint16, 6: np.int32, 7: np.uint32, 8: np.int64, 9: np.bool_,
               10: np.int8, 11: np.uint16}


def _shape(attr: Any) -> tuple[int, ...]:
    dims = getattr(attr, "dims", getattr(attr, "shape", ()))
    n = int(getattr(attr, "n_dims", len(dims)))
    return tuple(int(d) for d in list(dims)[:n])


def _cuda_code(result: Any) -> int:
    """cuda-python returns ``(error_enum, ...)`` for every cudart call."""
    if isinstance(result, tuple):
        result = result[0]
    try:
        return int(result)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"invalid CUDA return value {result!r}") from exc


class RKNNBackend:
    def __init__(self, model_path: str, seq_len: int, max_options: int, core_mask: str = "auto",
                 runtime: Any | None = None, metadata: dict[str, Any] | None = None):
        self._lock = threading.Lock()
        try:
            if runtime is None:
                from rknnlite.api import RKNNLite
                runtime = RKNNLite(verbose=False)
            self.runtime = runtime
            if runtime.load_rknn(model_path) != 0:
                raise RuntimeError("load_rknn failed")
            from rknnlite.api import RKNNLite
            cores = {"auto": getattr(RKNNLite, "NPU_CORE_AUTO", None),
                     "0": getattr(RKNNLite, "NPU_CORE_0", None),
                     "0_1": getattr(RKNNLite, "NPU_CORE_0_1", None),
                     "0_1_2": getattr(RKNNLite, "NPU_CORE_0_1_2", None)}
            if core_mask not in cores or cores[core_mask] is None:
                raise RuntimeError(f"unsupported RKNN core mask {core_mask!r}")
            if runtime.init_runtime(core_mask=cores[core_mask]) != 0:
                raise RuntimeError("init_runtime failed")
            rt = getattr(runtime, "rknn_runtime", runtime)
            try:
                sdk_version = rt.get_sdk_version()
                # rknnlite returns (banner, SDK version, driver version) on boards.
                # Health exposes the semantic SDK version; the raw banner is not a version.
                if isinstance(sdk_version, (tuple, list)) and len(sdk_version) > 1:
                    self.runtime_version = str(sdk_version[1])
                    self.driver_version = str(sdk_version[2]) if len(sdk_version) > 2 else None
                else:
                    self.runtime_version = str(sdk_version)
                    self.driver_version = None
            except Exception:
                self.runtime_version = None
                self.driver_version = None
            n_in, n_out = rt.get_in_out_num()
            self.inputs = [rt.get_tensor_attr(i, is_output=False) for i in range(n_in)]
            self.outputs = [rt.get_tensor_attr(i, is_output=True) for i in range(n_out)]
            names = [getattr(a, "name", b"") for a in self.inputs]
            names = [n.decode() if isinstance(n, bytes) else str(n) for n in names]
            expected_names = ["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"]
            if set(names) != set(expected_names):
                raise RuntimeError(f"RKNN model must expose five named inputs; got {names}")
            self.inputs_by_name = dict(zip(names, self.inputs))
            self.seq_len, self.max_options = seq_len, max_options
            self._check_shapes()
            md = metadata or {}
            if md.get("seq_len") is not None and int(md["seq_len"]) != seq_len:
                raise RuntimeError(f"model metadata seq_len {md['seq_len']} does not match configured {seq_len}")
            self.ready, self.reason = True, None
        except Exception as exc:
            try:
                if runtime is not None:
                    runtime.release()
            except Exception:
                pass
            self.ready, self.reason = False, str(exc)

    def _check_shapes(self):
        expected = {"input_ids": (1, self.seq_len), "attention_mask": (1, self.seq_len),
                    "marker_pos": (1, self.max_options), "marker_mask": (1, self.max_options),
                    "qtype": (1,)}
        for name, want in expected.items():
            attr = self.inputs_by_name[name]
            if _shape(attr) != want:
                raise RuntimeError(f"input shape {_shape(attr)} does not match {want}")

    def infer_request(self, request: EncodedRequest) -> list[float]:
        if not self.ready:
            raise NativeError("model_unavailable", self.reason or "RKNN unavailable")
        arrays = {"input_ids": request.input_ids, "attention_mask": request.attention_mask,
                  "marker_pos": request.marker_pos, "marker_mask": request.marker_mask,
                  "qtype": request.qtype}
        feeds = []
        for attr in self.inputs:
            name = getattr(attr, "name", b"")
            name = name.decode() if isinstance(name, bytes) else str(name)
            a = arrays[name]
            typ = int(getattr(attr, "type", 0))
            if typ not in _RKNN_DTYPE:
                raise NativeError("native_invalid", f"unmapped RKNN input dtype {typ}")
            feeds.append(np.ascontiguousarray(a.astype(_RKNN_DTYPE[typ], copy=False)))
        with self._lock:
            try:
                outputs = self.runtime.inference(inputs=feeds)
            except Exception as exc:
                raise NativeError("native_inference_failed", f"RKNN inference failed: {exc}") from exc
        choice = None
        for i, out in enumerate(outputs):
            attr = self.outputs[i] if i < len(self.outputs) else None
            name = getattr(attr, "name", b"") if attr is not None else b""
            name = name.decode() if isinstance(name, bytes) else str(name)
            a = np.asarray(out)
            lname = name.lower()
            if "act" in lname:
                continue
            if "choice" in lname or "logit" in lname or (choice is None and a.ndim >= 1 and a.shape[-1] == self.max_options):
                choice = a
        if choice is None:
            raise NativeError("native_invalid", "RKNN output has no choice logits")
        row = np.asarray(choice).reshape(-1)
        if len(row) != self.max_options:
            raise NativeError("native_invalid", f"choice logits must have {self.max_options} slots")
        active = np.asarray(request.marker_mask).reshape(-1)[:self.max_options].astype(bool)
        active_scores = row[active]
        if len(active_scores) != len(request.option_ids) or not np.isfinite(active_scores).all():
            raise NativeError("native_invalid", "active choice logits are missing or non-finite")
        return [float(x) for x in active_scores]

    def release(self):
        try:
            self.runtime.release()
        except Exception:
            pass


class TensorRTBackend:
    def __init__(self, model_path: str, seq_len: int, max_options: int, runtime: Any | None = None,
                 metadata: dict[str, Any] | None = None):
        self._lock = threading.Lock()
        self.logger = self.trt_runtime = self.engine = self.context = None
        try:
            import cuda.cudart as cudart
            import tensorrt as trt
            self.cudart, self.trt = cudart, trt
            self.runtime_version = str(getattr(trt, "__version__", "unknown"))
            if _cuda_code(cudart.cudaSetDevice(0)) != 0:
                raise RuntimeError("cudaSetDevice(0) failed")
            if runtime is None:
                with open(model_path, "rb") as f:
                    self.logger = trt.Logger(trt.Logger.WARNING)
                    self.trt_runtime = trt.Runtime(self.logger)
                    self.engine = self.trt_runtime.deserialize_cuda_engine(f.read())
            else:
                # Tests and callers may inject an already-created engine; vendor runtime
                # ownership remains explicit and is released only when we own it.
                self.engine = runtime
            if self.engine is None:
                raise RuntimeError("deserialize_cuda_engine returned no engine")
            self.context = self.engine.create_execution_context()
            self._setup_io(seq_len, max_options)
            if metadata and metadata.get("seq_len") is not None and int(metadata["seq_len"]) != seq_len:
                raise RuntimeError(f"model metadata seq_len {metadata['seq_len']} does not match configured {seq_len}")
            self.ready, self.reason = True, None
        except Exception as exc:
            self.release()
            self.ready, self.reason = False, str(exc)

    @staticmethod
    def _release_object(obj: Any) -> None:
        if obj is None:
            return
        for method in ("release", "destroy"):
            fn = getattr(obj, method, None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass
                break

    def release(self) -> None:
        """Release TensorRT resources in dependency order; safe during startup failure."""
        with self._lock:
            self._release_object(self.context)
            self._release_object(self.engine)
            self._release_object(self.trt_runtime)
            self._release_object(self.logger)
            self.context = self.engine = self.trt_runtime = self.logger = None
            self.ready = False

    def _setup_io(self, seq_len, max_options):
        self.io = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dt = self.engine.get_tensor_dtype(name)
            shape = tuple(int(x) for x in self.engine.get_tensor_shape(name))
            is_input = self.engine.get_tensor_mode(name) == self.trt.TensorIOMode.INPUT
            self.io.append((name, dt, shape, is_input))
        expected = {"input_ids": (1, seq_len), "attention_mask": (1, seq_len),
                    "marker_pos": (1, max_options), "marker_mask": (1, max_options), "qtype": (1,)}
        input_shapes = {name: shape for name, _, shape, is_input in self.io if is_input}
        if input_shapes != expected:
            raise RuntimeError(f"TensorRT inputs must be exactly {expected}; got {input_shapes}")

    def infer_request(self, request: EncodedRequest) -> list[float]:
        if not self.ready:
            raise NativeError("model_unavailable", self.reason or "TensorRT unavailable")
        np_of = {self.trt.DataType.FLOAT: np.float32, self.trt.DataType.HALF: np.float16,
                 self.trt.DataType.INT32: np.int32, self.trt.DataType.INT64: np.int64,
                 self.trt.DataType.BOOL: np.bool_, self.trt.DataType.INT8: np.int8,
                 self.trt.DataType.UINT8: np.uint8}
        arrays = {"input_ids": request.input_ids, "attention_mask": request.attention_mask,
                  "marker_pos": request.marker_pos, "marker_mask": request.marker_mask, "qtype": request.qtype}
        ptrs, outs = {}, {}
        try:
            with self._lock:
                for name, dt, shape, is_input in self.io:
                    nbytes = int(np.prod(shape)) * np.dtype(np_of[dt]).itemsize
                    err, ptr = self.cudart.cudaMalloc(nbytes)
                    if _cuda_code(err) != 0:
                        raise RuntimeError(f"cudaMalloc for {name} failed with CUDA error {err}")
                    ptrs[name] = ptr
                    self.context.set_tensor_address(name, ptr)
                    if is_input:
                        key = name if name in arrays else name.lower()
                        if key not in arrays:
                            raise RuntimeError(f"TensorRT input {name!r} is not one of the five native inputs")
                        a = np.ascontiguousarray(arrays[key].astype(np_of[dt], copy=False))
                        if tuple(a.shape) != shape:
                            raise RuntimeError(f"input {name} shape {a.shape} does not match engine {shape}")
                        err = self.cudart.cudaMemcpy(ptr, a.ctypes.data, nbytes,
                            self.cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)
                        if _cuda_code(err) != 0:
                            raise RuntimeError(f"cudaMemcpy H2D for {name} failed with CUDA error {err}")
                    else:
                        outs[name] = (dt, shape)
                self.cudart.cudaDeviceSynchronize()
                if not self.context.execute_async_v3(stream_handle=0):
                    raise RuntimeError("TensorRT execute_async_v3 returned false")
                err = self.cudart.cudaDeviceSynchronize()
                if _cuda_code(err) != 0:
                    raise RuntimeError(f"CUDA synchronize failed with CUDA error {err}")
                choice = None
                for name, (dt, shape) in outs.items():
                    buf = np.empty(int(np.prod(shape)), dtype=np_of[dt])
                    err = self.cudart.cudaMemcpy(buf.ctypes.data, ptrs[name], buf.nbytes,
                        self.cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
                    if _cuda_code(err) != 0:
                        raise RuntimeError(f"cudaMemcpy D2H for {name} failed with CUDA error {err}")
                    lname = name.lower()
                    if "act" in lname:
                        continue
                    if "choice" in lname or "logit" in lname or (choice is None and shape[-1] == len(request.marker_mask.reshape(-1))):
                        choice = buf
                if choice is None or len(choice) != len(request.marker_mask.reshape(-1)):
                    raise NativeError("native_invalid", "TensorRT choice logits must have max_options slots")
                active = np.asarray(request.marker_mask).reshape(-1).astype(bool)
                active_scores = choice[active]
                if len(active_scores) != len(request.option_ids) or not np.isfinite(active_scores).all():
                    raise NativeError("native_invalid", "active TensorRT choice logits are missing or non-finite")
                return [float(x) for x in active_scores]
        except NativeError:
            raise
        except Exception as exc:
            raise NativeError("native_inference_failed", f"TensorRT inference failed: {exc}") from exc
        finally:
            # Free under the same context lock used for address binding/execution.
            with self._lock:
                for ptr in ptrs.values():
                    try:
                        self.cudart.cudaFree(ptr)
                    except Exception:
                        pass
