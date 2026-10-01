"""laya_trt_run_cudart.py + CUDA Graph capture path.

Superset of the existing runner: with no --graph flag the code path is byte-for-byte the same
default-stream path the existing baseline numbers were produced with (execute_async_v3 on
stream 0), so graph/non-graph numbers are directly comparable.

With --graph:
  1. create a non-default stream (the legacy default stream cannot be captured)
  2. warm up on that stream, then cudaStreamBeginCapture / enqueueV3 / cudaStreamEndCapture
  3. cudaGraphInstantiate -> cudaGraphLaunch, timed around stream sync
Tensor addresses and shapes are set once before capture, so the graph needs no update per run.
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
ap.add_argument("--graph", action="store_true")
ap.add_argument("--graph-warmup", type=int, default=10)
args = ap.parse_args()

res = {"name": args.name, "mode": args.mode, "graph": bool(args.graph), "ok": False,
       "samples": [], "warnings": []}
graph_exec = None
graph_obj = None
capmode_used = None
stream = None


def _e(ret):
    """cudart wrappers return (err, ...) tuples; tolerate plain ints too."""
    return ret[0] if isinstance(ret, tuple) else ret


def _estr(e):
    try:
        r = cudart.cudaGetErrorString(e)
        if isinstance(r, tuple) and len(r) > 1:
            m = r[1]
            return m.decode() if isinstance(m, (bytes, bytearray)) else str(m)
        return str(r)
    except Exception as ex:
        return "<cudaGetErrorString unavailable: %s>" % ex


def ck(ret, what):
    e = _e(ret)
    if e != 0:
        raise RuntimeError("%s failed err=%s (%s)" % (what, e, _estr(e)))
    return ret


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
    # explicit precondition check: every dynamic dimension resolved to a concrete value means
    # the engine is fully static, i.e. the captured graph never needs a shape update.
    res["precondition"] = {
        "all_static_shapes": all(-1 not in s["shape"] for s in sig),
        "addresses_set_before_capture": all(ctx.get_tensor_address(s["name"]) == ptrs[s["name"]]
                                            for s in sig),
        "num_io_tensors": len(sig),
    }
    print("### [%s] bindings=%s" % (args.name, sig), flush=True)
    print("### [%s] precondition=%s" % (args.name, res["precondition"]), flush=True)

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

    if args.graph:
        err, stream = cudart.cudaStreamCreate()
        ck((err,), "cudaStreamCreate")
        res["stream"] = int(stream)

        def enqueue_once():
            """Pre-capture warmup: plain enqueueV3 on the capture stream, no graph yet."""
            ck(cudart.cudaStreamSynchronize(stream), "cudaStreamSynchronize")
            t0 = time.perf_counter()
            ok = ctx.execute_async_v3(stream_handle=stream)
            ck(cudart.cudaStreamSynchronize(stream), "cudaStreamSynchronize")
            if not ok:
                raise RuntimeError("execute_async_v3 returned False")
            return (time.perf_counter() - t0) * 1000.0

        def run_once():
            ck(cudart.cudaStreamSynchronize(stream), "cudaStreamSynchronize")
            t0 = time.perf_counter()
            ck(cudart.cudaGraphLaunch(graph_exec, stream), "cudaGraphLaunch")
            ck(cudart.cudaStreamSynchronize(stream), "cudaStreamSynchronize")
            return (time.perf_counter() - t0) * 1000.0
    else:
        def enqueue_once():
            return run_once()

        def run_once():
            cudart.cudaDeviceSynchronize()
            t0 = time.perf_counter()
            ok = ctx.execute_async_v3(stream_handle=0)
            cudart.cudaDeviceSynchronize()
            if not ok:
                raise RuntimeError("execute_async_v3 returned False")
            return (time.perf_counter() - t0) * 1000.0

    dirs = sorted(glob.glob(os.path.join(args.heldout, "sample_*")))
    print("### [%s] heldout=%s samples=%d" % (args.name, args.heldout, len(dirs)), flush=True)
    feed(dirs[0])
    for _ in range(args.warmup):
        enqueue_once()

    if args.graph:
        # capture. ThreadLocal first; Relaxed only if TRT touches another thread's stream.
        cap = None
        ok = ctx.execute_async_v3(stream_handle=stream)  # dry enqueue on target stream
        if not ok:
            raise RuntimeError("execute_async_v3 returned False on capture stream")
        ck(cudart.cudaStreamSynchronize(stream), "cudaStreamSynchronize")
        attempts = []
        for mode_name, mode in (("ThreadLocal", cudart.cudaStreamCaptureMode.cudaStreamCaptureModeThreadLocal),
                                ("Relaxed", cudart.cudaStreamCaptureMode.cudaStreamCaptureModeRelaxed)):
            try:
                ck(cudart.cudaStreamBeginCapture(stream, mode), "cudaStreamBeginCapture(%s)" % mode_name)
            except RuntimeError as e:
                attempts.append({"mode": mode_name, "stage": "BeginCapture", "error": str(e)})
                continue
            try:
                ok = ctx.execute_async_v3(stream_handle=stream)
                if not ok:
                    raise RuntimeError("execute_async_v3 returned False inside capture")
                r = cudart.cudaStreamEndCapture(stream)
                if _e(r) != 0 or len(r) < 2:
                    raise RuntimeError("cudaStreamEndCapture failed err=%s (%s) raw=%r"
                                       % (_e(r), _estr(_e(r)), r))
                cap = r[1]
                capmode_used = mode_name
                attempts.append({"mode": mode_name, "stage": "ok"})
                break
            except RuntimeError as e:
                attempts.append({"mode": mode_name, "stage": "capture", "error": str(e)})
                try:  # clear the invalidated capture state
                    cudart.cudaStreamEndCapture(stream)
                except Exception:
                    pass
        res["capture_attempts"] = attempts
        if cap is None:
            raise RuntimeError("CUDA graph capture failed in all modes: %s" % attempts)
        graph_obj = cap
        res["capture_mode"] = capmode_used

        # graph contents -- proves the whole network landed in the graph, not just part of it
        # binding form: cudaGraphGetNodes(graph, numNodes) -> (err, nodes, actual_count)
        try:
            r = cudart.cudaGraphGetNodes(graph_obj, 8192)
            if _e(r) == 0:
                res["graph_nodes"] = int(r[2]) if len(r) > 2 else len(r[1])
            else:
                res["graph_nodes"] = None
                res["graph_nodes_error"] = _estr(_e(r))
        except Exception as ex:
            res["graph_nodes"] = None
            res["graph_nodes_error"] = "%s: %s" % (type(ex).__name__, ex)
        try:
            r2 = cudart.cudaGraphGetRootNodes(graph_obj, 8192)
            if _e(r2) == 0:
                res["graph_root_nodes"] = int(r2[2]) if len(r2) > 2 else len(r2[1])
            else:
                res["graph_root_nodes"] = None
                res["root_nodes_error"] = _estr(_e(r2))
        except Exception as ex:
            res["graph_root_nodes"] = None
            res["root_nodes_error"] = "%s: %s" % (type(ex).__name__, ex)

        t0 = time.perf_counter()
        ri, inst_form, inst_errs = None, None, []
        for form, fn in (("(graph, flags=0)", lambda: cudart.cudaGraphInstantiate(graph_obj, 0)),
                         ("(graph)", lambda: cudart.cudaGraphInstantiate(graph_obj)),
                         ("(graph, b'', 0)", lambda: cudart.cudaGraphInstantiate(graph_obj, b"", 0))):
            try:
                t0 = time.perf_counter()
                r = fn()
                inst_ms = (time.perf_counter() - t0) * 1000.0
                ri, inst_form = r, form
                break
            except TypeError as ex:
                inst_errs.append({"form": form, "error": str(ex)})
        res["instantiate_form"] = inst_form
        res["instantiate_sig_attempts"] = inst_errs
        if ri is None:
            raise RuntimeError("no usable cudaGraphInstantiate signature: %s" % inst_errs)
        res["instantiate_ret_arity"] = len(ri) if isinstance(ri, tuple) else 1
        if _e(ri) != 0:
            raise RuntimeError("cudaGraphInstantiate failed err=%s (%s) raw=%r"
                               % (_e(ri), _estr(_e(ri)), ri))
        if not isinstance(ri, tuple) or len(ri) < 2:
            raise RuntimeError("cudaGraphInstantiate returned unexpected %r" % (ri,))
        graph_exec = ri[1]
        res["instantiate_err_node"] = str(ri[2]) if len(ri) > 2 else None
        res["instantiate_ms"] = round(inst_ms, 3)
        print("### [%s] capture OK mode=%s nodes=%s roots=%s instantiate_ms=%.3f"
              % (args.name, capmode_used, res["graph_nodes"], res["graph_root_nodes"], inst_ms),
              flush=True)

        # extra warmup on the instantiated graph, as required
        for _ in range(args.graph_warmup):
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
            if args.graph:
                ck(cudart.cudaStreamSynchronize(stream), "cudaStreamSynchronize")
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
