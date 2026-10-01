#!/usr/bin/env python3
"""单进程 RKNN 探针：load -> init -> inference（带 golden 对照），结果写 JSON。

设计约束（来自 device-gotchas #7）：
  - rknnlite 的 inference() 可能直接 SIGSEGV，所以每个 artifact 必须独立子进程跑，
    父进程只能拿到 return code / stderr tail。
  - load_rknn() / init_runtime() 成功都不构成"能跑"的证据，只有 inference() 成功、
    输出 finite 且 shape 正确才算。

实测确认的 rknnlite 2.3.0 API（radxa, py3.11）：
  rt.get_in_out_num() / rt.get_tensor_attr(i, is_output) -> .name .type .dims .fmt

输入 dtype 适配：ONNX 侧是 int64 / bool，RKNN 侧几乎必然是 int32 / uint8，
所以喂给 inference() 的数组必须按 RKNN 自报的 dtype 转，不能直接用 golden 的数组。

用法：
  python3 probe_laya_rknn.py --rknn model.rknn --golden-dir ./golden \
      --out result.json [--core-mask auto|0|0_1|0_1_2] [--iters 20] [--warmup 3]
"""
import argparse
import json
import os
import sys
import time
import traceback

import numpy as np


CORE_MASKS = ("auto", "0", "0_1", "0_1_2")

RKNN_TYPE_MAP = {
    0: np.float32,
    1: np.float16,
    2: np.int8,
    3: np.uint8,
    4: np.int16,
    5: np.uint16,
    6: np.int32,
    7: np.uint32,
    8: np.int64,
    9: np.bool_,
    10: np.int8,      # INT4：无 numpy 原生类型，零填充场景用 int8 承载
    11: np.uint16,    # BFLOAT16：无 numpy 原生类型，零填充场景用 uint16 承载
}
# ⚠️ 枚举顺序见 rknn_api.h `_rknn_tensor_type`。**缺键会静默兜底成 float32**
#    （.get(type, np.float32)），给 int64 输入喂 float32 数组会让 set_inputs
#    缓冲区越界 -> 无诊断信息的 SIGSEGV。本模型实测输入类型是 8(INT64) 与 9(BOOL)，
#    早期版本的表里没有这两项，就是崩溃的真正原因。

ONNX_DTYPE_MAP = {
    "float32": np.float32, "FLOAT": np.float32,
    "float16": np.float16, "FLOAT16": np.float16,
    "int8": np.int8, "INT8": np.int8,
    "uint8": np.uint8, "UINT8": np.uint8,
    "int16": np.int16, "INT16": np.int16,
    "int32": np.int32, "INT32": np.int32,
    "int64": np.int64, "INT64": np.int64,
    "bool": np.bool_, "BOOL": np.bool_,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rknn", required=True)
    ap.add_argument("--golden-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--core-mask", default="auto", choices=CORE_MASKS)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()

    result = {
        "rknn": os.path.abspath(args.rknn),
        "rknn_size_mb": round(os.path.getsize(args.rknn) / 1048576.0, 2),
        "core_mask": args.core_mask,
        "ok": False,
        "stage": "start",
        "adaptations": [],
        "warnings": [],
    }

    try:
        from rknnlite.api import RKNNLite
    except Exception as e:
        result["stage"] = "import"
        result["error"] = "rknnlite import failed: %r" % (e,)
        _write(args.out, result); return 1

    rknn = RKNNLite()
    try:
        result["stage"] = "load"
        t0 = time.perf_counter()
        ret = rknn.load_rknn(args.rknn)
        result["load_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        result["load_ret"] = int(ret)
        if ret != 0:
            result["error"] = "load_rknn returned %s" % ret
            _write(args.out, result); return 2

        result["stage"] = "init"
        core = {"auto": RKNNLite.NPU_CORE_AUTO, "0": RKNNLite.NPU_CORE_0,
                "0_1": RKNNLite.NPU_CORE_0_1, "0_1_2": RKNNLite.NPU_CORE_0_1_2}[args.core_mask]
        t0 = time.perf_counter()
        ret = rknn.init_runtime(core_mask=core)
        result["init_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        result["init_ret"] = int(ret)
        if ret != 0:
            result["error"] = "init_runtime returned %s" % ret
            _write(args.out, result); return 3

        rt = rknn.rknn_runtime
        n_in, n_out = rt.get_in_out_num()
        inputs_sig = []
        for i in range(n_in):
            a = rt.get_tensor_attr(i, is_output=False)
            # dims 是补齐到 16 元素的定长数组，尾部补 0，必须按 n_dims 截断
            nd = int(a.n_dims)
            dims = [int(d) for d in a.dims][:nd] if nd > 0 else [int(d) for d in a.dims if int(d)]
            inputs_sig.append({
                "index": i,
                "name": a.name.decode() if isinstance(a.name, bytes) else str(a.name),
                "type": int(a.type),
                "dtype": str(RKNN_TYPE_MAP.get(int(a.type), np.float32)),
                "dims": dims,
                "n_dims": nd,
                "fmt": int(a.fmt),
                "pass_through": int(a.pass_through),
            })
        result["rknn_inputs"] = inputs_sig
        result["sdk_version"] = str(rt.get_sdk_version())

        # ---- golden ----
        result["stage"] = "prepare"
        with open(os.path.join(args.golden_dir, "io_spec.json")) as f:
            spec = json.load(f)
        by_name = {t["name"]: t for t in inputs_sig}

        feed = []
        for item in spec["inputs"]:
            name = item["name"]
            path = os.path.join(args.golden_dir, name + ".npy")
            if not os.path.exists(path):
                raise FileNotFoundError("missing golden input: " + path)
            arr = np.load(path)

            rk = by_name.get(name)
            if rk is None:
                result["warnings"].append("golden input %s not present in rknn inputs" % name)
                feed.append(arr)
                continue
            want = RKNN_TYPE_MAP.get(rk["type"], np.float32)
            if arr.dtype != want:
                conv = arr.astype(want)
                note = "%s: golden %s -> rknn %s (shape %s -> %s)" % (
                    name, arr.dtype, want, list(arr.shape), rk["dims"])
                if tuple(arr.shape) != tuple(rk["dims"]):
                    note += "  !! SHAPE MISMATCH"
                    result["warnings"].append(note)
                else:
                    result["adaptations"].append(note)
                arr = conv
            elif tuple(arr.shape) != tuple(rk["dims"]):
                result["warnings"].append(
                    "%s: shape mismatch golden=%s rknn=%s" % (name, list(arr.shape), rk["dims"]))
            feed.append(arr)

        # ---- inference ----
        result["stage"] = "inference"
        for _ in range(args.warmup):
            rknn.inference(inputs=feed)

        lat, outs = [], None
        for _ in range(args.iters):
            t0 = time.perf_counter()
            outs = rknn.inference(inputs=feed)
            lat.append((time.perf_counter() - t0) * 1000)

        if outs is None:
            result["error"] = "inference returned None"
            _write(args.out, result); return 4

        ls = sorted(lat); n = len(ls)
        result["latency_ms"] = {
            "n": n, "p50": round(ls[n // 2], 2),
            "p95": round(ls[min(n - 1, int(n * 0.95))], 2),
            "min": round(ls[0], 2), "max": round(ls[-1], 2),
            "mean": round(sum(lat) / n, 2),
        }

        # ---- compare vs golden ----
        result["stage"] = "compare"
        summary = []
        for item, out in zip(spec["outputs"], outs):
            arr = np.asarray(out)
            entry = {
                "name": item["name"],
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "finite": bool(np.isfinite(arr).all()),
                "n_nan": int(np.isnan(arr).sum()) if arr.dtype.kind == "f" else 0,
                "n_inf": int(np.isinf(arr).sum()) if arr.dtype.kind == "f" else 0,
                "maxabs": float(np.abs(arr).max()) if arr.size else 0.0,
                "first8": [float(v) for v in arr.reshape(-1)[:8]],
            }
            gp = os.path.join(args.golden_dir, item["name"] + ".npy")
            if os.path.exists(gp):
                g = np.load(gp)
                if g.shape == arr.shape:
                    gf = g.astype(np.float64)
                    af = arr.astype(np.float64)
                    denom = np.linalg.norm(gf)
                    entry["rel_l2"] = float(np.linalg.norm(af - gf) / denom) if denom else None
                    entry["maxabsdiff"] = float(np.abs(af - gf).max())
                else:
                    entry["compare_error"] = "shape mismatch golden=%s rknn=%s" % (
                        list(g.shape), list(arr.shape))
            summary.append(entry)
        result["outputs"] = summary
        result["ok"] = True
        _write(args.out, result); return 0

    except BaseException as e:
        result["error"] = "%s: %s" % (type(e).__name__, e)
        result["traceback"] = traceback.format_exc()[-2000:]
        _write(args.out, result); return 5
    finally:
        try:
            rknn.release()
        except Exception:
            pass


def _write(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    sys.exit(main())
