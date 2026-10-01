"""Run a TensorRT engine over the laya heldout rows: dump logits/act_logits + latency stats.

Mirrors scripts/47_device_dump.py (the RKNN dumper) so the dump JSON can be fed straight into
scripts/46_rank_from_dumps.py on the host.

  --mode rank  : feed the 16 heldout samples once each, dump logits (latency = one-shot per sample)
  --mode bench : feed sample 0 --iters times, report p50/p95/min/max/mean over the loop
"""
import argparse
import glob
import json
import os
import sys
import time
import traceback

import numpy as np
import cupy as cp
import tensorrt as trt

LOG = trt.Logger(trt.Logger.WARNING)

NP_OF = {
    trt.DataType.FLOAT: np.float32,
    trt.DataType.HALF: np.float16,
    trt.DataType.INT32: np.int32,
    trt.DataType.INT64: np.int64,
    trt.DataType.BOOL: np.bool_,
    trt.DataType.INT8: np.int8,
    trt.DataType.UINT8: np.uint8,
    trt.DataType.BF16: None,
    trt.DataType.FP8: None,
}


def load_engine(path):
    with open(path, "rb") as f:
        blob = f.read()
    rt = trt.Runtime(LOG)
    eng = rt.deserialize_cuda_engine(blob)
    if eng is None:
        raise RuntimeError("deserialize_cuda_engine returned None")
    return eng


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--mode", default="rank", choices=["rank", "bench"])
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    args = ap.parse_args()

    res = {"name": args.name, "engine": args.engine, "mode": args.mode, "ok": False,
           "samples": [], "warnings": []}
    try:
        res["engine_size_mb"] = round(os.path.getsize(args.engine) / 1048576.0, 2)
        eng = load_engine(args.engine)
        ctx = eng.create_execution_context()
        res["trt_version"] = trt.__version__

        names = [eng.get_tensor_name(i) for i in range(eng.num_io_tensors)]
        sig = []
        bufs = {}
        outs = []
        for n in names:
            mode = eng.get_tensor_mode(n)
            is_in = (mode == trt.TensorIOMode.INPUT)
            dt = eng.get_tensor_dtype(n)
            shp = tuple(int(d) for d in eng.get_tensor_shape(n))
            sig.append({"name": n, "mode": str(mode), "is_input": is_in, "dtype": str(dt),
                        "shape": list(shp)})
            if dt not in NP_OF or NP_OF[dt] is None:
                raise RuntimeError("unsupported binding dtype %s for %s" % (dt, n))
            arr = cp.empty(shp, dtype=NP_OF[dt])
            bufs[n] = arr
            ctx.set_tensor_address(n, arr.data.ptr)
            if mode == trt.TensorIOMode.OUTPUT:
                outs.append(n)
        res["bindings"] = sig
        print("### [%s] bindings=%s" % (args.name, sig), flush=True)

        stream = cp.cuda.Stream(non_blocking=True)

        def run_once():
            stream.synchronize()
            t0 = time.perf_counter()
            if not ctx.execute_async_v3(stream_handle=stream.ptr):
                raise RuntimeError("execute_async_v3 returned False")
            stream.synchronize()
            return (time.perf_counter() - t0) * 1000.0

        def feed(sample_dir):
            for s in sig:
                if not s["is_input"]:
                    continue
                p = os.path.join(sample_dir, s["name"] + ".npy")
                if not os.path.exists(p):
                    raise SystemExit("missing %s in %s" % (s["name"] + ".npy", sample_dir))
                a = np.load(p)
                want = NP_OF[eng.get_tensor_dtype(s["name"])]
                if list(a.shape) != s["shape"]:
                    res["warnings"].append("shape %s vs engine %s for %s"
                                           % (list(a.shape), s["shape"], s["name"]))
                    a = a.reshape(s["shape"])
                bufs[s["name"]][...] = cp.asarray(a.astype(want))

        dirs = sorted(glob.glob(os.path.join(args.heldout, "sample_*")))
        print("### [%s] heldout=%s samples=%d" % (args.name, args.heldout, len(dirs)), flush=True)
        if not dirs:
            raise SystemExit("no sample_* under %s" % args.heldout)

        feed(dirs[0])
        for _ in range(args.warmup):
            run_once()

        if args.mode == "bench":
            lat = [run_once() for _ in range(args.iters)]
            ls = sorted(lat)
            def pct(q):
                return round(ls[min(len(ls) - 1, int(q * len(ls)))], 3)
            res["latency_ms"] = {"n": len(ls), "p50": pct(0.50), "p95": pct(0.95),
                                 "min": round(ls[0], 3), "max": round(ls[-1], 3),
                                 "mean": round(sum(ls) / len(ls), 3)}
            print("### [%s] bench n=%d %s" % (args.name, len(ls), res["latency_ms"]), flush=True)
            res["latency_raw_ms"] = [round(v, 3) for v in lat]
        else:
            lat = []
            for k, d in enumerate(dirs):
                feed(d)
                lat.append(run_once())
                row = {"idx": k}
                for n in outs:
                    row[n] = [float(v) for v in cp.asnumpy(bufs[n]).reshape(-1)]
                if "logits" not in row and len(row) == 2:
                    res["warnings"].append("no 'logits' output name; outputs=%s" % outs)
                res["samples"].append(row)
                print("### [%s] sim[%02d] %s" % (args.name, k, row), flush=True)
            ls = sorted(lat)
            res["latency_ms"] = {"n": len(ls), "p50": round(ls[len(ls) // 2], 3),
                                 "p95": round(ls[min(len(ls) - 1, int(0.95 * len(ls)))], 3),
                                 "min": round(ls[0], 3), "max": round(ls[-1], 3),
                                 "mean": round(sum(lat) / len(lat), 3)}
            res["latency_raw_ms"] = [round(v, 3) for v in lat]
            print("### [%s] latency(1 shot per sample) %s" % (args.name, res["latency_ms"]), flush=True)

        res["ok"] = True
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
        print("### wrote", args.out, flush=True)
        return 0
    except BaseException as e:
        res["error"] = "%s: %s" % (type(e).__name__, e)
        res["traceback"] = traceback.format_exc()[-2000:]
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
        print("### ERROR %s" % res["error"], flush=True)
        print(res["traceback"], flush=True)
        return 5


if __name__ == "__main__":
    sys.exit(main())
