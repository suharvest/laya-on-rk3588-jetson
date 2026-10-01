"""Per-layer real device timing for one .rknn, via RKNNRuntime.get_run_perf (RKNN_QUERY_PERF_DETAIL).

Feed construction and dtype maps are imported from 47_device_dump.py, not re-typed, so this probe
feeds the model exactly like the existing latency/accuracy scripts do.

Times one inference per held-out row (or `--iters` cycled calls), then asks the runtime for the
per-layer breakdown of the last run.
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
ap.add_argument("--iters", type=int, default=16)
ap.add_argument("--warmup", type=int, default=3)
ap.add_argument("--core-mask", default="0_1_2", choices=sorted(_m.CORE))
a = ap.parse_args()


def w(o, p):
    with open(p, "w") as f:
        json.dump(o, f, indent=2)


res = {"name": a.name, "rknn": a.rknn, "core_mask": a.core_mask, "ok": False,
       "iters": a.iters, "warmup": a.warmup}
out = res
try:
    r = RKNNLite(verbose=False)
    if r.load_rknn(a.rknn) != 0:
        out["error"] = "load_rknn failed"; w(out, a.out); print("### LOAD_FAILED"); raise SystemExit(2)
    if r.init_runtime(core_mask=_m.CORE[a.core_mask]) != 0:
        out["error"] = "init_runtime failed"; w(out, a.out); print("### INIT_FAILED"); raise SystemExit(3)

    rt = r.rknn_runtime
    n_in, _n_out = rt.get_in_out_num()
    sig = []
    for i in range(n_in):
        at = rt.get_tensor_attr(i, is_output=False)
        sig.append({"index": i,
                    "name": at.name.decode() if isinstance(at.name, bytes) else str(at.name),
                    "type": int(at.type), "dtype": _m.RKNN_TYPE.get(int(at.type)),
                    "dims": [int(d) for d in at.dims][:int(at.n_dims)]})
    out["inputs"] = sig
    out["sdk_version"] = str(rt.get_sdk_version())
    print("### [%s] sdk=%s inputs=%s" % (a.name, out["sdk_version"], json.dumps(sig)), flush=True)

    dirs = sorted(glob.glob(os.path.join(a.heldout, "sample_*")))
    print("### [%s] heldout=%s samples=%d" % (a.name, a.heldout, len(dirs)), flush=True)
    assert len(dirs) == 16, "expected 16 held-out rows, got %d" % len(dirs)

    def feed_for(d):
        o = []
        for s in sig:
            t = _m.NP_DTYPE[s["dtype"]]
            o.append(np.load(os.path.join(d, s["name"] + ".npy")).astype(t))
        return o

    feeds = [feed_for(d) for d in dirs]

    for i in range(a.warmup):
        r.inference(inputs=feeds[i % len(feeds)])

    lat = []
    for i in range(a.iters):
        k = i % len(feeds)
        t0 = time.perf_counter()
        r.inference(inputs=feeds[k])
        lat.append((time.perf_counter() - t0) * 1000.0)
    s = sorted(lat)
    out["p50"] = round(s[len(s) // 2], 2)
    out["p95"] = round(s[min(len(s) - 1, int(0.95 * len(s)))], 2)
    out["min"] = round(s[0], 2)
    out["max"] = round(s[-1], 2)
    out["latency_ms_all"] = [round(v, 2) for v in lat]
    print("### [%s] p50=%.2f p95=%.2f min=%.2f max=%.2f iters=%d"
          % (a.name, out["p50"], out["p95"], out["min"], out["max"], a.iters), flush=True)

    # one untimed pass over row 0 immediately before reading the perf detail
    r.inference(inputs=feeds[0])

    raw = None
    try:
        raw = rt.get_run_perf()
    except Exception as e:  # noqa: BLE001
        out["get_run_perf_error"] = repr(e)
        print("### [%s] get_run_perf raised: %r" % (a.name, e), flush=True)

    def norm(x):
        if x is None:
            return None
        if isinstance(x, bytes):
            return x.decode("utf-8", "replace")
        if isinstance(x, (list, tuple)):
            return [norm(y) for y in x]
        return str(x)

    out["perf_raw_type"] = type(raw).__name__
    out["perf_raw_len"] = len(raw) if hasattr(raw, "__len__") else None
    nr = norm(raw)
    out["perf_raw"] = nr
    print("### [%s] PERF_RAW type=%s" % (a.name, out["perf_raw_type"]), flush=True)
    if isinstance(nr, list):
        for ln in nr:
            print("PERFLINE|%s" % ln, flush=True)
    elif nr is not None:
        for ln in str(nr).splitlines():
            print("PERFLINE|%s" % ln, flush=True)

    out["ok"] = True
except Exception as e:  # noqa: BLE001
    import traceback
    out["error"] = repr(e)
    out["traceback"] = traceback.format_exc()
    print("### [%s] EXCEPTION %r" % (a.name, e), flush=True)
    print(traceback.format_exc(), flush=True)

w(out, a.out)
print("### PERFLAYER_DONE %s ok=%s" % (a.name, out.get("ok")), flush=True)
