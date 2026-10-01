"""One model, one process: run an .rknn on the device over the held-out rows and dump outputs.

Two layouts are supported so the same dumper covers the 5-input fp16 model and the single-input
packed model:
  five   -- sample dir holds input_ids.npy / attention_mask.npy / marker_pos.npy /
            marker_mask.npy / qtype.npy
  auto   -- feed each RKNN input by its own name from the sample dir; if the model wants 'packed'
            and the dir has no packed.npy, build it from the five files in the fixed order
            [input_ids(90) | attention_mask(90) | marker_pos(16) | marker_mask(16) | qtype(1)].

Input dtypes come from the model itself through get_tensor_attr(); the rknnlite type enum is the
one in rknn_api.h (8 = INT64, 9 = BOOL). An unmapped type is a hard error: defaulting an unknown
key to float32 is what silently overran the buffer and SIGSEGV'd in an earlier probe.
"""
import argparse
import glob
import json
import os
import sys
import time
import traceback

import numpy as np
from rknnlite.api import RKNNLite

FIVE = ["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"]
SPANS = {"input_ids": (0, 90), "attention_mask": (90, 180), "marker_pos": (180, 196),
         "marker_mask": (196, 212), "qtype": (212, 213)}
PLEN = 213

NP_DTYPE = {
    "INT64": np.int64, "INT32": np.int32, "INT16": np.int16, "INT8": np.int8,
    "UINT8": np.uint8, "UINT16": np.uint16, "UINT32": np.uint32,
    "BOOL": np.bool_, "FLOAT32": np.float32, "FLOAT16": np.float16, "FLOAT64": np.float64,
}
RKNN_TYPE = {0: "FLOAT32", 1: "FLOAT16", 2: "INT8", 3: "UINT8", 4: "INT16", 5: "UINT16",
             6: "INT32", 7: "UINT32", 8: "INT64", 9: "BOOL", 10: "INT8", 11: "UINT16"}

CORE = {"auto": RKNNLite.NPU_CORE_AUTO, "0": RKNNLite.NPU_CORE_0, "0_1": RKNNLite.NPU_CORE_0_1,
        "0_1_2": RKNNLite.NPU_CORE_0_1_2}


def pack_from_five(d):
    parts = []
    for n in FIVE:
        a = np.load(os.path.join(d, n + ".npy"))
        if n == "marker_mask":
            a = a.astype(np.int64)
        parts.append(np.asarray(a, dtype=np.int64).reshape(-1))
    return np.concatenate(parts).reshape(1, PLEN).astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rknn", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--core-mask", default="0_1_2", choices=sorted(CORE))
    ap.add_argument("--iters", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=1)
    args = ap.parse_args()

    res = {"name": args.name, "rknn": args.rknn, "core_mask": args.core_mask, "ok": False,
           "samples": [], "warnings": []}
    r = RKNNLite(verbose=False)
    try:
        res["rknn_size_mb"] = round(os.path.getsize(args.rknn) / 1048576.0, 2)
        if r.load_rknn(args.rknn) != 0:
            res["error"] = "load_rknn failed"
            _w(args.out, res); return 2
        if r.init_runtime(core_mask=CORE[args.core_mask]) != 0:
            res["error"] = "init_runtime failed"
            _w(args.out, res); return 3
        rt = r.rknn_runtime
        n_in, n_out = rt.get_in_out_num()
        sig = []
        for i in range(n_in):
            a = rt.get_tensor_attr(i, is_output=False)
            nd = int(a.n_dims)
            sig.append({"index": i,
                        "name": a.name.decode() if isinstance(a.name, bytes) else str(a.name),
                        "type": int(a.type), "dtype": RKNN_TYPE.get(int(a.type)),
                        "dims": [int(d) for d in a.dims][:nd]})
        res["inputs"] = sig
        res["sdk_version"] = str(rt.get_sdk_version())
        print("### [%s] inputs=%s" % (args.name, sig), flush=True)

        dirs = sorted(glob.glob(os.path.join(args.heldout, "sample_*")))
        print("### [%s] heldout=%s samples=%d" % (args.name, args.heldout, len(dirs)), flush=True)

        def feed_for(d):
            feeds = []
            for s in sig:
                t = NP_DTYPE.get(s["dtype"])
                if t is None:
                    raise SystemExit("unmapped rknn input type %r (%s)" % (s["type"], s["dtype"]))
                p = os.path.join(d, s["name"] + ".npy")
                if os.path.exists(p):
                    a = np.load(p)
                elif s["name"] == "packed":
                    a = pack_from_five(d)
                    res["warnings"].append("built packed from five inputs for %s" % os.path.basename(d))
                else:
                    raise SystemExit("no %s.npy in %s" % (s["name"], d))
                a = a.astype(t)
                if list(a.shape) != s["dims"]:
                    res["warnings"].append("shape %s vs rknn %s for %s" % (list(a.shape), s["dims"], s["name"]))
                feeds.append(a)
            return feeds

        for _ in range(args.warmup):
            r.inference(inputs=feed_for(dirs[0]))

        lat = []
        for k, d in enumerate(dirs):
            feed = feed_for(d)
            tt = time.perf_counter()
            outs = r.inference(inputs=feed)
            lat.append((time.perf_counter() - tt) * 1000)
            lg = ac = None
            for o in outs:
                a = np.asarray(o)
                if a.ndim == 2 and a.shape[-1] == 2:
                    ac = a
                else:
                    lg = a
            row = {"idx": k, "logits": [float(v) for v in np.asarray(lg).reshape(-1)],
                   "act_logits": [float(v) for v in np.asarray(ac).reshape(-1)]}
            res["samples"].append(row)
            print("### [%s] sim[%02d] logits=%s act=%s" % (args.name, k, row["logits"], row["act_logits"]),
                  flush=True)
        ls = sorted(lat)
        res["latency_ms"] = {"n": len(ls), "p50": round(ls[len(ls) // 2], 2), "min": round(ls[0], 2),
                             "max": round(ls[-1], 2), "mean": round(sum(ls) / len(ls), 2)}
        res["ok"] = True
        print("### [%s] latency(16 samples, one shot each) %s" % (args.name, res["latency_ms"]), flush=True)
        _w(args.out, res)
        return 0
    except BaseException as e:
        res["error"] = "%s: %s" % (type(e).__name__, e)
        res["traceback"] = traceback.format_exc()[-1500:]
        _w(args.out, res)
        return 5
    finally:
        try:
            r.release()
        except Exception:
            pass


def _w(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


if __name__ == "__main__":
    sys.exit(main())
