"""Same as laya_trt_run.py but moves buffers with cudaMemcpy (cuda-python) instead of cupy
elementwise kernels: on orin-nx cupy's JIT setitem kernel fails with CUDA_ERROR_INVALID_IMAGE.
"""
import argparse
import glob
import json
import os
import time
import traceback

import numpy as np
import cuda.cudart as cudart
import tensorrt as trt

LOG = trt.Logger(trt.Logger.WARNING)
NP_OF = {trt.DataType.FLOAT: np.float32, trt.DataType.HALF: np.float16,
         trt.DataType.INT32: np.int32, trt.DataType.INT64: np.int64,
         trt.DataType.BOOL: np.bool_, trt.DataType.INT8: np.int8, trt.DataType.UINT8: np.uint8}

ap = argparse.ArgumentParser()
ap.add_argument("--engine", required=True)
ap.add_argument("--heldout", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--mode", default="rank", choices=["rank", "bench"])
ap.add_argument("--iters", type=int, default=200)
ap.add_argument("--warmup", type=int, default=20)
args = ap.parse_args()

res = {"name": args.name, "mode": args.mode, "ok": False, "samples": [], "warnings": []}
try:
    res["engine_size_mb"] = round(os.path.getsize(args.engine) / 1048576.0, 2)
    with open(args.engine, "rb") as f:
        blob = f.read()
    eng = trt.Runtime(LOG).deserialize_cuda_engine(blob)
    ctx = eng.create_execution_context()
    res["trt_version"] = trt.__version__

    sig, ptrs, outs = [], {}, []
    for i in range(eng.num_io_tensors):
        n = eng.get_tensor_name(i)
        dt = eng.get_tensor_dtype(n)
        shp = tuple(int(d) for d in eng.get_tensor_shape(n))
        is_in = eng.get_tensor_mode(n) == trt.TensorIOMode.INPUT
        nbytes = int(np.prod(shp)) * np.dtype(NP_OF[dt]).itemsize
        err, p = cudart.cudaMalloc(nbytes)
        if err != 0:
            raise RuntimeError("cudaMalloc(%d) failed err=%d" % (nbytes, err))
        ptrs[n] = p
        sig.append({"name": n, "is_input": is_in, "dtype": str(dt), "shape": list(shp),
                    "nbytes": nbytes})
        ctx.set_tensor_address(n, p)
        if not is_in:
            outs.append(n)
    res["bindings"] = sig
    print("### [%s] bindings=%s" % (args.name, sig), flush=True)

    def run_once():
        cudart.cudaDeviceSynchronize()
        t0 = time.perf_counter()
        ok = ctx.execute_async_v3(stream_handle=0)
        cudart.cudaDeviceSynchronize()
        if not ok:
            raise RuntimeError("execute_async_v3 returned False")
        return (time.perf_counter() - t0) * 1000.0

    def feed(d):
        for s in sig:
            if not s["is_input"]:
                continue
            p = os.path.join(d, s["name"] + ".npy")
            a = np.load(p).astype(NP_OF[eng.get_tensor_dtype(s["name"])]).reshape(-1)
            if list(np.load(p).shape) != s["shape"]:
                res["warnings"].append("shape %s vs engine %s for %s"
                                       % (list(np.load(p).shape), s["shape"], s["name"]))
            a = np.ascontiguousarray(a)
            cudart.cudaMemcpy(ptrs[s["name"]], a.ctypes.data, s["nbytes"],
                              cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

    def read(n, dtype):
        buf = np.empty(int(np.prod(eng.get_tensor_shape(n))), dtype=dtype)
        cudart.cudaMemcpy(buf.ctypes.data, ptrs[n], buf.nbytes,
                          cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
        return buf.reshape(tuple(int(d) for d in eng.get_tensor_shape(n)))

    dirs = sorted(glob.glob(os.path.join(args.heldout, "sample_*")))
    print("### [%s] heldout=%s samples=%d" % (args.name, args.heldout, len(dirs)), flush=True)
    feed(dirs[0])
    for _ in range(args.warmup):
        run_once()

    if args.mode == "bench":
        lat = [run_once() for _ in range(args.iters)]
        ls = sorted(lat)
        res["latency_ms"] = {"n": len(ls), "p50": round(ls[len(ls) // 2], 3),
                             "p95": round(ls[min(len(ls) - 1, int(0.95 * len(ls)))], 3),
                             "min": round(ls[0], 3), "max": round(ls[-1], 3),
                             "mean": round(sum(lat) / len(lat), 3)}
        print("### [%s] bench %s" % (args.name, res["latency_ms"]), flush=True)
        res["latency_raw_ms"] = [round(v, 3) for v in lat]
    else:
        lat = []
        for k, d in enumerate(dirs):
            feed(d)
            lat.append(run_once())
            row = {"idx": k}
            for n in outs:
                row[n] = [float(v) for v in read(n, NP_OF[eng.get_tensor_dtype(n)]).reshape(-1)]
            res["samples"].append(row)
            print("### [%s] sim[%02d] %s" % (args.name, k, row), flush=True)
        ls = sorted(lat)
        res["latency_ms"] = {"n": len(ls), "p50": round(ls[len(ls) // 2], 3),
                             "p95": round(ls[min(len(ls) - 1, int(0.95 * len(ls)))], 3),
                             "min": round(ls[0], 3), "max": round(ls[-1], 3),
                             "mean": round(sum(lat) / len(lat), 3)}
        res["latency_raw_ms"] = [round(v, 3) for v in lat]
        print("### [%s] latency %s" % (args.name, res["latency_ms"]), flush=True)
    res["ok"] = True
except BaseException as e:
    res["error"] = "%s: %s" % (type(e).__name__, e)
    res["traceback"] = traceback.format_exc()[-2500:]
    print("### ERROR %s" % res["error"], flush=True)
    print(res["traceback"], flush=True)

with open(args.out, "w") as f:
    json.dump(res, f, indent=2)
print("### wrote", args.out, flush=True)
