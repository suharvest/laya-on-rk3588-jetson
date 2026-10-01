#!/usr/bin/env python3
"""Same comparison as v2_device_three.py, but resolves the 4-D input's layout empirically.

RKNN declares the ONNX mask input [1,12,512,512] as [1,512,512,12] (fmt=1, NHWC). Three different
byte layouts can satisfy that declared shape, and feeding the wrong one silently scrambles the mask
(all-zeros masks are layout-invariant, so only a non-trivial mask exposes the mistake).

Rather than guess, run the candidate layouts and let ORT pick the winner. The UNFUSED model is the
calibration: it is a faithful lowering, so the layout that reproduces ORT there is the correct
convention, and the fused model is then read under that same convention.

  layouts: asis  -- feed [1,12,512,512] raw (buffer reinterpreted as [1,512,512,12])
           t231  -- mask[0,c,i,j] -> x[0,i,j,c]
           t321  -- mask[0,c,j,i] -> x[0,i,j,c]

RKNN_TYPE_MAP is imported verbatim from probe_laya_rknn.py -- not modified.
"""
import argparse
import json
import os
import sys
import time
import traceback

import numpy as np
from rknnlite.api import RKNNLite

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_laya_rknn import RKNN_TYPE_MAP  # noqa: E402

CORE = {"auto": RKNNLite.NPU_CORE_AUTO, "0": RKNNLite.NPU_CORE_0, "0_1": RKNNLite.NPU_CORE_0_1,
        "0_1_2": RKNNLite.NPU_CORE_0_1_2}
LAYOUTS = {"asis": lambda a: a,
           "t231": lambda a: np.transpose(a, (0, 2, 3, 1)),
           "t321": lambda a: np.transpose(a, (0, 3, 2, 1))}


def stats(a):
    a = np.asarray(a).astype(np.float64)
    return {"shape": list(a.shape), "min": float(a.min()), "max": float(a.max()),
            "mean": float(a.mean()), "absmean": float(np.abs(a).mean()),
            "absmax": float(np.abs(a).max()), "std": float(a.std()),
            "finite": bool(np.isfinite(a).all()),
            "n_nan": int(np.isnan(a).sum()), "n_inf": int(np.isinf(a).sum())}


def cmp(a, b):
    a = np.asarray(a).astype(np.float64)
    b = np.asarray(b).astype(np.float64)
    d = a - b
    nb = np.linalg.norm(b)
    return {"max_abs_diff": float(np.abs(d).max()),
            "mean_abs_diff": float(np.abs(d).mean()),
            "rmse": float(np.sqrt((d * d).mean())),
            "rel_l2": float(np.linalg.norm(d) / nb) if nb else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rknn", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--ort", required=True)
    ap.add_argument("--sets", default="mask0,maskpad")
    ap.add_argument("--layouts", default="asis,t231,t321")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump-dir", default="")
    ap.add_argument("--core-mask", default="0_1_2", choices=sorted(CORE))
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()

    res = {"label": args.label, "rknn": os.path.abspath(args.rknn),
           "rknn_size_mb": round(os.path.getsize(args.rknn) / 1048576.0, 2),
           "core_mask": args.core_mask, "ok": False, "runs": {}, "warnings": []}
    r = RKNNLite(verbose=False)
    try:
        if r.load_rknn(args.rknn) != 0:
            res["error"] = "load_rknn failed"; _w(args.out, res); return 2
        if r.init_runtime(core_mask=CORE[args.core_mask]) != 0:
            res["error"] = "init_runtime failed"; _w(args.out, res); return 3
        rt = r.rknn_runtime
        n_in, _ = rt.get_in_out_num()
        sig = []
        for i in range(n_in):
            a = rt.get_tensor_attr(i, is_output=False)
            nd = int(a.n_dims)
            sig.append({"index": i,
                        "name": a.name.decode() if isinstance(a.name, bytes) else str(a.name),
                        "type": int(a.type), "dims": [int(d) for d in a.dims][:nd],
                        "fmt": int(a.fmt)})
        res["inputs"] = sig
        res["sdk_version"] = str(rt.get_sdk_version())
        print("### [%s] sig=%s" % (args.label, json.dumps(sig)), flush=True)

        by_name = {}
        for s in sig:
            by_name["hidden" if len(s["dims"]) == 3 else "mask"] = s
        for k, s in by_name.items():
            res.setdefault("mapping", {})[k] = s["name"]

        for set_name in args.sets.split(","):
            ms = by_name["mask"]
            want = RKNN_TYPE_MAP.get(ms["type"])
            if want is None:
                raise SystemExit("unmapped rknn input type %r" % (ms["type"],))
            m_src = np.load(os.path.join(args.inputs, set_name, "mask.npy"))
            h_src = np.load(os.path.join(args.inputs, set_name, "hidden.npy"))
            hs = by_name["hidden"]
            hwant = RKNN_TYPE_MAP.get(hs["type"])
            if tuple(h_src.shape) != tuple(hs["dims"]):
                res["warnings"].append("hidden shape %s vs rknn %s" % (list(h_src.shape), hs["dims"]))
            gp = os.path.join(args.ort, args.label, set_name, "out.npy")
            g = np.load(gp) if os.path.exists(gp) else None

            for lname in args.layouts.split(","):
                key = "%s|%s" % (set_name, lname)
                e = {}
                try:
                    m_arr = LAYOUTS[lname](m_src)
                    e["mask_fed_shape"] = list(m_arr.shape)
                    e["mask_fed_dtype"] = str(want)
                    e["mask_shape_ok"] = list(m_arr.shape) == ms["dims"]
                    feed = [h_src.astype(hwant), m_arr.astype(want)]
                    # keep feed ordered by rknn input index
                    order = sorted(range(len(sig)), key=lambda i: sig[i]["index"])
                    named = {s["name"]: f for s, f in zip(sig, feed)}
                    feed = [named[s["name"]] for s in sig]
                    for _ in range(args.warmup):
                        r.inference(inputs=feed)
                    lat = []
                    for _ in range(args.iters):
                        t0 = time.perf_counter()
                        outs = r.inference(inputs=feed)
                        lat.append((time.perf_counter() - t0) * 1000.0)
                    o = np.asarray(outs[0])
                    e["dev"] = stats(o)
                    if g is not None and g.shape == o.shape:
                        e["ort"] = stats(g)
                        e["vs_ort"] = cmp(g, o)
                    if args.dump_dir and lname == "t231":
                        dd = os.path.join(args.dump_dir, args.label, set_name)
                        os.makedirs(dd, exist_ok=True)
                        np.save(os.path.join(dd, "out.npy"), o)
                    ls = sorted(lat)
                    e["latency_ms"] = {"n": len(ls), "p50": round(ls[len(ls) // 2], 3),
                                       "min": round(ls[0], 3), "max": round(ls[-1], 3)}
                except BaseException as ex:
                    e["run_error"] = "%s: %s" % (type(ex).__name__, ex)
                    e["traceback"] = traceback.format_exc()[-800:]
                res["runs"][key] = e
                print("### [%s][%s] %s" % (args.label, key, json.dumps(e)), flush=True)
        res["ok"] = True
        _w(args.out, res); return 0
    except BaseException as e:
        res["error"] = "%s: %s" % (type(e).__name__, e)
        res["traceback"] = traceback.format_exc()[-2000:]
        _w(args.out, res); return 5
    finally:
        try:
            r.release()
        except Exception:
            pass


def _w(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    sys.exit(main())
