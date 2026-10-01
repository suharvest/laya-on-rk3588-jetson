#!/usr/bin/env python3
"""Run one .rknn on the device and compare against the ORT fp32 reference for the SAME onnx.

The three minimal variants do NOT share weights (80_mkvariants.py builds a fresh Blk per mode with
the seed set only once before the loop), so the only valid comparison is
    RKNN(X).rknn  vs  ORT(X.onnx)
i.e. same graph, fused-by-RKNN vs not. Comparing RKNN(noguard) against RKNN(guard) or ORT(guard)
would be comparing two different models.

RKNN_TYPE_MAP is imported verbatim from probe_laya_rknn.py -- not modified, not duplicated.

  python3 v2_device_three.py --rknn m.rknn --label var-noguard \
      --inputs ./inputs --ort ./ort --sets mask0,maskpad --out ./r_noguard.json
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
from probe_laya_rknn import RKNN_TYPE_MAP  # noqa: E402  (verbatim, unmodified)

CORE = {"auto": RKNNLite.NPU_CORE_AUTO, "0": RKNNLite.NPU_CORE_0, "0_1": RKNNLite.NPU_CORE_0_1,
        "0_1_2": RKNNLite.NPU_CORE_0_1_2}


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
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump-dir", default="")
    ap.add_argument("--core-mask", default="0_1_2", choices=sorted(CORE))
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()

    res = {"label": args.label, "rknn": os.path.abspath(args.rknn),
           "rknn_size_mb": round(os.path.getsize(args.rknn) / 1048576.0, 2),
           "core_mask": args.core_mask, "ok": False, "sets": {}, "warnings": []}
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

        # map sig entries to the 2 canonical input names by declared dims (mask is the 4-D one)
        by_name = {}
        for s in sig:
            if len(s["dims"]) == 3:
                by_name["hidden"] = s
            elif len(s["dims"]) == 4:
                by_name["mask"] = s
            else:
                raise SystemExit("unexpected rknn input arity %s" % s)
        print("### [%s] name mapping=%s" % (args.label, json.dumps({k: v["name"] for k, v in by_name.items()})), flush=True)

        for set_name in args.sets.split(","):
            e = {}
            feed = []
            for name, s in by_name.items():
                p = os.path.join(args.inputs, set_name, name + ".npy")
                arr = np.load(p)
                want = RKNN_TYPE_MAP.get(s["type"])
                if want is None:
                    raise SystemExit("unmapped rknn input type %r" % (s["type"],))
                e["feed_%s" % name] = {"src_dtype": str(arr.dtype), "src_shape": list(arr.shape),
                                       "rknn_type": s["type"], "rknn_dtype": str(want),
                                       "rknn_dims": s["dims"]}
                if tuple(arr.shape) != tuple(s["dims"]):
                    e["feed_%s" % name]["SHAPE_MISMATCH"] = True
                    res["warnings"].append("%s shape %s vs rknn %s" % (name, list(arr.shape), s["dims"]))
                feed.append(arr.astype(want))

            for _ in range(args.warmup):
                r.inference(inputs=feed)
            lat, outs = [], None
            for _ in range(args.iters):
                t0 = time.perf_counter()
                outs = r.inference(inputs=feed)
                lat.append((time.perf_counter() - t0) * 1000.0)
            if outs is None:
                e["error"] = "inference returned None"
                res["sets"][set_name] = e
                continue
            o = np.asarray(outs[0])
            e["dev"] = stats(o)
            if args.dump_dir:
                dd = os.path.join(args.dump_dir, args.label, set_name)
                os.makedirs(dd, exist_ok=True)
                np.save(os.path.join(dd, "out.npy"), o)
                e["dumped"] = dd
            gp = os.path.join(args.ort, args.label, set_name, "out.npy")
            if os.path.exists(gp):
                g = np.load(gp)
                if g.shape == o.shape:
                    e["ort"] = stats(g)
                    e["vs_ort"] = cmp(g, o)      # (reference=ORT, test=device)
                else:
                    e["compare_error"] = "shape ORT %s vs dev %s" % (list(g.shape), list(o.shape))
            else:
                e["compare_error"] = "missing ORT ref %s" % gp
            ls = sorted(lat)
            e["latency_ms"] = {"n": len(ls), "p50": round(ls[len(ls) // 2], 3),
                               "min": round(ls[0], 3), "max": round(ls[-1], 3),
                               "mean": round(sum(lat) / len(lat), 3)}
            res["sets"][set_name] = e
            print("### [%s][%s] %s" % (args.label, set_name, json.dumps(e)), flush=True)
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
