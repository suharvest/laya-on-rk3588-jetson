#!/usr/bin/env python3
"""通用 RKNN 冒烟探针：load -> init -> 读输入签名 -> 喂零输入 -> inference。

用途：不依赖 golden，快速回答"这个体积的 .rknn 在这块板子上能不能真的跑"。
必须在独立子进程里跑（rknnlite 的 inference() 可能直接 SIGSEGV）。

实测确认的 rknnlite 2.3.0 API（radxa, py3.11）：
  rt = rknn.rknn_runtime
  rt.get_in_out_num()                     -> (n_input, n_output)
  rt.get_tensor_attr(i, is_output=False)  -> attr: .name(bytes) .type .dims .n_dims
                                                    .n_elems .size .fmt .pass_through
  注意：没有 input_attrs / model 属性，shape 叫 dims。

用法：
  python3 npu_size_probe.py --rknn X.rknn --out X.json [--core-mask 0_1_2] [--iters 10]
"""
import argparse
import json
import os
import sys
import time
import traceback

import numpy as np


CORE_MASKS = ("auto", "0", "0_1", "0_1_2")

# RKNN tensor attr 的 type 编码 -> numpy dtype。
# 枚举定义见 rknn_api.h `_rknn_tensor_type`（顺序即值）：
#   0 FLOAT32 / 1 FLOAT16 / 2 INT8 / 3 UINT8 / 4 INT16 / 5 UINT16 /
#   6 INT32 / 7 UINT32 / 8 INT64 / 9 BOOL / 10 INT4 / 11 BFLOAT16
# ⚠️ 曾经把 6 当成 INT64、7 当成 BOOL（错位两格），导致对 INT32 输入喂 int64 数组，
#    set_inputs 缓冲区越界 -> 静默 SIGSEGV。改这里之前先对一遍上面的枚举。
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


def read_signature(rt, n_input, n_output):
    def one(i, is_out):
        a = rt.get_tensor_attr(i, is_output=is_out)
        name = a.name.decode() if isinstance(a.name, bytes) else str(a.name)
        # 实测：dims 是补齐到 16 个元素的定长数组，尾部补 0，必须按 n_dims 截断，
        # 否则会构造出含 0 维的数组，inference() 静默返回 None。
        nd = int(a.n_dims)
        dims = [int(d) for d in a.dims][:nd] if nd > 0 else [int(d) for d in a.dims if int(d)]
        return {
            "index": i,
            "name": name,
            "type": int(a.type),
            "dtype": str(RKNN_TYPE_MAP.get(int(a.type), np.float32)),
            "dims": dims,
            "n_dims": nd,
            "n_elems": int(a.n_elems),
            "fmt": int(a.fmt),
            "pass_through": int(a.pass_through),
        }
    return ([one(i, False) for i in range(n_input)],
            [one(i, True) for i in range(n_output)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rknn", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--core-mask", default="0_1_2", choices=CORE_MASKS)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2)
    args = ap.parse_args()

    res = {
        "rknn": os.path.abspath(args.rknn),
        "size_mb": round(os.path.getsize(args.rknn) / 1048576.0, 2),
        "core_mask": args.core_mask,
        "ok": False,
        "stage": "start",
    }

    from rknnlite.api import RKNNLite

    rknn = RKNNLite()
    try:
        res["stage"] = "load"
        t0 = time.perf_counter()
        ret = rknn.load_rknn(args.rknn)
        res["load_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        res["load_ret"] = int(ret)
        if ret != 0:
            res["error"] = "load_rknn=%s" % ret
            _w(args.out, res); return 2

        res["stage"] = "init"
        core = {"auto": RKNNLite.NPU_CORE_AUTO, "0": RKNNLite.NPU_CORE_0,
                "0_1": RKNNLite.NPU_CORE_0_1, "0_1_2": RKNNLite.NPU_CORE_0_1_2}[args.core_mask]
        t0 = time.perf_counter()
        ret = rknn.init_runtime(core_mask=core)
        res["init_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        res["init_ret"] = int(ret)
        if ret != 0:
            res["error"] = "init_runtime=%s" % ret
            _w(args.out, res); return 3

        rt = rknn.rknn_runtime
        res["stage"] = "signature"
        n_in, n_out = rt.get_in_out_num()
        res["n_in"], res["n_out"] = int(n_in), int(n_out)
        ins, outs_sig = read_signature(rt, n_in, n_out)
        res["inputs"] = ins
        res["outputs_sig"] = outs_sig
        res["sdk_version"] = str(rt.get_sdk_version())

        feed = [np.zeros(tuple(t["dims"]), dtype=RKNN_TYPE_MAP.get(t["type"], np.float32))
                for t in ins]

        res["stage"] = "inference"
        for _ in range(args.warmup):
            rknn.inference(inputs=feed)
        lat, outs = [], None
        for _ in range(args.iters):
            t0 = time.perf_counter()
            outs = rknn.inference(inputs=feed)
            lat.append((time.perf_counter() - t0) * 1000)

        if outs is None:
            res["error"] = "inference returned None"
            _w(args.out, res); return 4

        ls = sorted(lat); n = len(ls)
        res["latency_ms"] = {"n": n, "p50": round(ls[n // 2], 1),
                             "p95": round(ls[min(n - 1, int(n * 0.95))], 1),
                             "min": round(ls[0], 1), "max": round(ls[-1], 1)}

        info = []
        for i, o in enumerate(outs):
            a = np.asarray(o)
            info.append({
                "index": i, "shape": list(a.shape), "dtype": str(a.dtype),
                "finite": bool(np.isfinite(a).all()),
                "n_nan": int(np.isnan(a).sum()) if a.dtype.kind == "f" else 0,
                "n_inf": int(np.isinf(a).sum()) if a.dtype.kind == "f" else 0,
                "maxabs": float(np.abs(a).max()) if a.size else 0.0,
            })
        res["outputs"] = info
        res["ok"] = True
        _w(args.out, res); return 0

    except BaseException as e:
        res["error"] = "%s: %s" % (type(e).__name__, e)
        res["traceback"] = traceback.format_exc()[-1500:]
        _w(args.out, res); return 5
    finally:
        try:
            rknn.release()
        except Exception:
            pass


def _w(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print(json.dumps(obj, indent=2, ensure_ascii=False)[:4000])


if __name__ == "__main__":
    sys.exit(main())
