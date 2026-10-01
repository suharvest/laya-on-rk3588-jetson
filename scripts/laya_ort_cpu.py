"""ONNX Runtime CPU baseline for laya s90, same heldout rows as the TensorRT runs."""
import argparse
import glob
import json
import os
import time

import numpy as np
import onnxruntime as ort

ap = argparse.ArgumentParser()
ap.add_argument("--onnx", required=True)
ap.add_argument("--heldout", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--iters", type=int, default=20)
ap.add_argument("--warmup", type=int, default=2)
ap.add_argument("--threads", type=int, default=6)
args = ap.parse_args()

so = ort.SessionOptions()
so.intra_op_num_threads = args.threads
so.inter_op_num_threads = 1
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
so.log_severity_level = 3

t0 = time.perf_counter()
sess = ort.InferenceSession(args.onnx, so, providers=["CPUExecutionProvider"])
load_s = time.perf_counter() - t0

ins = sess.get_inputs()
sig = [{"name": i.name, "type": i.type, "shape": i.shape} for i in ins]
print("### ort", ort.__version__, "providers", sess.get_providers(), flush=True)
print("### inputs", sig, flush=True)
print("### session load %.2f s" % load_s, flush=True)

d0 = sorted(glob.glob(os.path.join(args.heldout, "sample_*")))[0]


def feed(d):
    f = {}
    for i in ins:
        a = np.load(os.path.join(d, i.name + ".npy"))
        t = np.bool_ if "bool" in i.type else (np.int64 if "int64" in i.type else np.float32)
        f[i.name] = a.astype(t)
    return f


f0 = feed(d0)
for _ in range(args.warmup):
    sess.run(None, f0)
lat = []
for _ in range(args.iters):
    t = time.perf_counter()
    sess.run(None, f0)
    lat.append((time.perf_counter() - t) * 1000)
ls = sorted(lat)
res = {"ort": ort.__version__, "inputs": sig, "threads": args.threads,
       "session_load_s": round(load_s, 2), "n": len(ls),
       "p50_ms": round(ls[len(ls) // 2], 2),
       "p95_ms": round(ls[min(len(ls) - 1, int(0.95 * len(ls)))], 2),
       "min_ms": round(ls[0], 2), "max_ms": round(ls[-1], 2),
       "mean_ms": round(sum(lat) / len(lat), 2),
       "raw_ms": [round(v, 2) for v in lat]}
print("### ORT_CPU", json.dumps({k: v for k, v in res.items() if k != "raw_ms"}), flush=True)
with open(args.out, "w") as f:
    json.dump(res, f, indent=2)
print("### wrote", args.out, flush=True)
