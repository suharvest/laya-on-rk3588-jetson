#!/usr/bin/env python3
"""同一台 RK3588 上的 CPU 基线：用 onnxruntime 跑同一份 ONNX 和同一组 golden 输入。

与 probe_laya_rknn.py 配对使用：同一个 golden-dir、同一个 io_spec.json，
唯一的差别是走 CPU 还是走 NPU。这样两者的延迟可以直接比（同机同输入同 batch）。

用法：
  python3 bench_laya_cpu.py --onnx model.onnx --golden-dir ./golden \
      --out cpu.json [--iters 20] [--warmup 3] [--threads 4]
"""
import argparse
import json
import os
import platform
import time

import numpy as np


DTYPE_MAP = {
    "float32": np.float32,
    "float16": np.float16,
    "int8": np.int8,
    "uint8": np.uint8,
    "int16": np.int16,
    "int32": np.int32,
    "int64": np.int64,
    "bool": np.bool_,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--golden-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--threads", type=int, default=0, help="0 = ORT 默认")
    args = ap.parse_args()

    import onnxruntime as ort

    so = ort.SessionOptions()
    if args.threads:
        so.intra_op_num_threads = args.threads
        so.inter_op_num_threads = 1
    sess = ort.InferenceSession(args.onnx, so, providers=["CPUExecutionProvider"])

    result = {
        "onnx": os.path.abspath(args.onnx),
        "onnx_size_mb": round(os.path.getsize(args.onnx) / 1048576.0, 2),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "ort_version": ort.__version__,
        "providers": sess.get_providers(),
        "threads": args.threads or "default",
        "ok": False,
    }

    with open(os.path.join(args.golden_dir, "io_spec.json")) as f:
        spec = json.load(f)

    feed = {}
    for item in spec["inputs"]:
        arr = np.load(os.path.join(args.golden_dir, item["name"] + ".npy"))
        feed[item["name"]] = arr

    for _ in range(args.warmup):
        sess.run(None, feed)

    lat = []
    outs = None
    for _ in range(args.iters):
        t0 = time.perf_counter()
        outs = sess.run(None, feed)
        lat.append((time.perf_counter() - t0) * 1000)

    lat_sorted = sorted(lat)
    n = len(lat_sorted)
    result["latency_ms"] = {
        "n": n,
        "p50": round(lat_sorted[n // 2], 2),
        "p95": round(lat_sorted[min(n - 1, int(n * 0.95))], 2),
        "min": round(lat_sorted[0], 2),
        "max": round(lat_sorted[-1], 2),
        "mean": round(sum(lat) / n, 2),
    }

    summary = []
    for item, out in zip(spec["outputs"], outs):
        arr = np.asarray(out)
        entry = {
            "name": item["name"],
            "shape": list(arr.shape),
            "maxabs": float(np.abs(arr).max()) if arr.size else 0.0,
            "first8": [float(v) for v in arr.reshape(-1)[:8]],
        }
        golden_path = os.path.join(args.golden_dir, item["name"] + ".npy")
        if os.path.exists(golden_path):
            g = np.load(golden_path).astype(np.float64)
            a = arr.astype(np.float64)
            if g.shape == a.shape:
                entry["rel_l2_vs_golden"] = float(
                    np.linalg.norm(a - g) / np.linalg.norm(g)) if np.linalg.norm(g) else None
                entry["maxabsdiff_vs_golden"] = float(np.abs(a - g).max())
        summary.append(entry)
    result["outputs"] = summary
    result["ok"] = True

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
