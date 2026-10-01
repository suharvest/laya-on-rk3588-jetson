"""Steady-state latency + per-sample dump for one .rknn on the device.

47_device_dump.py declares --iters but never uses it: it times exactly the 16 held-out rows once
each, which is the n=16 single-shot regime whose run-to-run spread (~25 ms on this model) is large
enough to flip conclusions. This script keeps the same feed construction -- the dtype maps and the
core-mask enum are imported from 47_device_dump.py itself, not re-typed -- and instead cycles the
16 rows for iters>=30 timed calls after a warmup, reporting p50/p95 over that sample.

Logits are taken from the final full pass over the 16 rows, so one process produces both the
ranking payload and the latency distribution.
"""
import argparse
import glob
import importlib.util
import json
import os
import time

import numpy as np
from rknnlite.api import RKNNLite

DUMPER = "/home/radxa/laya-rknn/47_device_dump.py"
_spec = importlib.util.spec_from_file_location("dump47", DUMPER)
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)

ap = argparse.ArgumentParser()
ap.add_argument("--rknn", required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--heldout", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--iters", type=int, default=48)
ap.add_argument("--warmup", type=int, default=5)
ap.add_argument("--core-mask", default="0_1_2", choices=sorted(_m.CORE))
a = ap.parse_args()

r = RKNNLite(verbose=False)
res = {"name": a.name, "rknn": a.rknn, "core_mask": a.core_mask, "ok": False,
       "iters": a.iters, "warmup": a.warmup, "samples": []}


def w(o, p):
    with open(p, "w") as f:
        json.dump(o, f, indent=2)


if r.load_rknn(a.rknn) != 0:
    res["error"] = "load_rknn failed"; w(res, a.out); print("### LOAD_FAILED"); raise SystemExit(2)
if r.init_runtime(core_mask=_m.CORE[a.core_mask]) != 0:
    res["error"] = "init_runtime failed"; w(res, a.out); print("### INIT_FAILED"); raise SystemExit(3)

rt = r.rknn_runtime
n_in, _ = rt.get_in_out_num()
sig = []
for i in range(n_in):
    at = rt.get_tensor_attr(i, is_output=False)
    sig.append({"index": i,
                "name": at.name.decode() if isinstance(at.name, bytes) else str(at.name),
                "type": int(at.type), "dtype": _m.RKNN_TYPE.get(int(at.type)),
                "dims": [int(d) for d in at.dims][:int(at.n_dims)]})
res["inputs"] = sig
res["sdk_version"] = str(rt.get_sdk_version())
print("### [%s] sdk=%s inputs=%s" % (a.name, res["sdk_version"], json.dumps(sig)), flush=True)

dirs = sorted(glob.glob(os.path.join(a.heldout, "sample_*")))
print("### [%s] heldout=%s samples=%d" % (a.name, a.heldout, len(dirs)), flush=True)
assert len(dirs) == 16, "expected 16 held-out rows, got %d" % len(dirs)


def feed_for(d):
    out = []
    for s in sig:
        t = _m.NP_DTYPE[s["dtype"]]
        arr = np.load(os.path.join(d, s["name"] + ".npy")).astype(t)
        out.append(arr)
    return out


feeds = [feed_for(d) for d in dirs]

for i in range(a.warmup):
    r.inference(inputs=feeds[i % len(feeds)])

lat = []
last = {}
for i in range(a.iters):
    k = i % len(feeds)
    t0 = time.perf_counter()
    outs = r.inference(inputs=feeds[k])
    lat.append((time.perf_counter() - t0) * 1000.0)
    last[k] = outs

s = sorted(lat)
res["n_latency"] = len(s)
res["p50"] = round(s[len(s) // 2], 2)
res["p95"] = round(s[min(len(s) - 1, int(0.95 * len(s)))], 2)
res["min"] = round(s[0], 2)
res["max"] = round(s[-1], 2)
res["mean"] = round(sum(s) / len(s), 2)
res["latency_ms_all"] = [round(v, 2) for v in lat]

for k in sorted(last):
    lg = ac = None
    for o in last[k]:
        arr = np.asarray(o)
        if arr.ndim == 2 and arr.shape[-1] == 2:
            ac = arr
        else:
            lg = arr
    res["samples"].append({"idx": k,
                           "logits": [float(v) for v in np.asarray(lg).reshape(-1)],
                           "act_logits": [float(v) for v in np.asarray(ac).reshape(-1)]})
res["ok"] = True
w(res, a.out)
print("### LAT %s iters=%d n=%d p50=%.2f p95=%.2f min=%.2f max=%.2f mean=%.2f"
      % (a.name, a.iters, res["n_latency"], res["p50"], res["p95"], res["min"], res["max"], res["mean"]),
      flush=True)
print("### LATENCY_DONE %s" % a.name, flush=True)
