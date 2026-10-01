"""RKNN converter with quantization + export head check + optional simulator rank dump.

Same construction as scripts/21_convert_quant.py (config kwargs, verbose log, build args), with
two additions:

  * the export is head-checked in-process right away (file size vs the size the header itself
    declares at offset 16), because export_rknn() returns 0 even when the file is truncated by
    a full disk;
  * --rank-dump runs the freshly built model on the simulator in the same session and writes the
    per-sample logits, so int8/w4a16 can be ranked against fp16 without a second build. The
    simulator cannot load an .rknn from disk, it can only run a graph built in this session.
"""
import argparse
import glob
import json
import os
import struct
import sys
import time
import traceback

from rknn.api import RKNN

RKNN_TYPE_MAP = {
    0: "float32", 1: "float16", 2: "int8", 3: "uint8", 4: "int16", 5: "uint16",
    6: "int32", 7: "uint32", 8: "int64", 9: "bool", 10: "int8", 11: "uint16",
}

ap = argparse.ArgumentParser()
ap.add_argument("--onnx", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--verbose-log", required=True)
ap.add_argument("--target", default="rk3588")
ap.add_argument("--opt-level", type=int, default=3, help="-1 omits optimization_level")
ap.add_argument("--do-quantization", action="store_true")
ap.add_argument("--quantized-dtype", default=None)
ap.add_argument("--quantized-algorithm", default=None)
ap.add_argument("--dataset", default=None, help="dataset txt: one .npy path per line")
ap.add_argument("--label", default="")
ap.add_argument("--rank-heldout", default=None,
                help="dir of sample_XXXX/ subdirs; each holds one .npy per model input")
ap.add_argument("--input-names", default="input_ids,attention_mask,marker_pos,marker_mask,qtype")
ap.add_argument("--rank-dump", default=None, help="json path for the simulator outputs")
ap.add_argument("--rank-iters", type=int, default=1)
ap.add_argument("--continue-on-bad-headcheck", action="store_true",
                help="do NOT stop when the export head check fails: the artifact is still "
                     "invalid, but the in-session simulator run is kept for diagnosis")
args = ap.parse_args()

ONNX, OUT = args.onnx, args.out
IN_NAMES = [n for n in args.input_names.split(",") if n]
os.makedirs(os.path.dirname(OUT), exist_ok=True)

if args.do_quantization and not args.dataset:
    print("### ARG_FAILED: --do-quantization needs --dataset", flush=True)
    sys.exit(8)

cfg = {"target_platform": args.target}
if args.opt_level >= 0:
    cfg["optimization_level"] = args.opt_level
if args.quantized_dtype:
    cfg["quantized_dtype"] = args.quantized_dtype
if args.quantized_algorithm:
    cfg["quantized_algorithm"] = args.quantized_algorithm

import rknn as _r
print("### label:", args.label, flush=True)
print("### rknn-toolkit2 version:", getattr(_r, "__version__", "n/a"), flush=True)
try:
    import importlib.metadata as md
    print("### importlib.metadata version:", md.version("rknn-toolkit2"), flush=True)
except Exception as e:
    print("### metadata version lookup failed:", e, flush=True)

print("### ONNX path:", ONNX, "size:", os.path.getsize(ONNX), flush=True)
print("### OUT path:", OUT, flush=True)
print("### CONFIG KWARGS:", json.dumps(cfg, sort_keys=True), flush=True)
print("### do_quantization:", args.do_quantization, "dataset:", args.dataset, flush=True)
if args.dataset:
    with open(args.dataset) as f:
        lines = [x for x in f.read().split("\n") if x.strip()]
    print("### dataset lines: %d, first=%s" % (len(lines), lines[0] if lines else None), flush=True)

t0 = time.time()
os.makedirs(os.path.dirname(args.verbose_log), exist_ok=True)
rknn_obj = RKNN(verbose=True, verbose_file=args.verbose_log)
print("### verbose_file:", args.verbose_log, flush=True)


def head_check(path):
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        f.seek(16)
        off16 = struct.unpack("<Q", f.read(8))[0]
    delta = size - off16
    ok = off16 > 0 and 0 <= delta <= 262144
    print("### HEADCHECK %s size=%d off16=%d delta=%d -> %s"
          % (path, size, off16, delta, "OK" if ok else "TRUNCATED_OR_BAD"), flush=True)
    return ok, size, off16, delta


try:
    ret = rknn_obj.config(**cfg)
    print("### config ret =", ret, flush=True)
    if ret != 0:
        print("### CONFIG_FAILED", flush=True)
        sys.exit(9)

    ret = rknn_obj.load_onnx(model=ONNX)
    print("### load_onnx ret =", ret, "elapsed=%.1fs" % (time.time() - t0), flush=True)
    if ret != 0:
        print("### LOAD_FAILED", flush=True)
        sys.exit(10)

    try:
        print("### toolkit-visible inputs:", rknn_obj.rknn_base.net._get_inputs(), flush=True)
    except Exception as e:
        print("### input introspection n/a:", e, flush=True)

    t1 = time.time()
    ret = rknn_obj.build(do_quantization=args.do_quantization,
                         dataset=args.dataset if args.do_quantization else None)
    print("### build ret =", ret, "elapsed=%.1fs" % (time.time() - t1), flush=True)
    if ret != 0:
        print("### BUILD_FAILED", flush=True)
        sys.exit(11)

    ret = rknn_obj.export_rknn(OUT)
    print("### export_rknn ret =", ret, flush=True)
    if ret != 0:
        print("### EXPORT_FAILED", flush=True)
        sys.exit(12)

    ok, size, off16, delta = head_check(OUT)
    if not ok:
        if not args.continue_on_bad_headcheck:
            print("### HEADCHECK_FAILED -- stopping before any further use of this artifact", flush=True)
            sys.exit(13)
        print("### HEADCHECK_FAILED_BUT_CONTINUING: this artifact is INVALID "
              "(size=%d off16=%d delta=%d); the simulator run below is diagnostic only"
              % (size, off16, delta), flush=True)

    # ---- optional simulator run on the held-out rows ------------------------------
    if args.rank_dump:
        try:
            t2 = time.time()
            ret = rknn_obj.init_runtime()
            print("### init_runtime (simulator) ret =", ret, flush=True)
            if ret != 0:
                raise RuntimeError("init_runtime returned %s" % ret)
            try:
                attrs = rknn_obj.rknn_runtime.get_tensor_attr(0, is_output=False)
                print("### sim input0 attr: name=%s type=%s" % (attrs.name, attrs.type), flush=True)
            except Exception as e:
                print("### sim attr introspection n/a:", e, flush=True)

            dirs = sorted(glob.glob(os.path.join(args.rank_heldout, "sample_*")))
            print("### rank-heldout %s samples=%d" % (args.rank_heldout, len(dirs)), flush=True)
            import numpy as np
            samples = []
            for d in dirs:
                samples.append([np.load(os.path.join(d, n + ".npy")) for n in IN_NAMES])
            print("### feed[0] shapes=%s dtypes=%s"
                  % ([list(a.shape) for a in samples[0]], [str(a.dtype) for a in samples[0]]), flush=True)
            for _ in range(max(0, args.rank_iters - 1)):
                rknn_obj.inference(inputs=samples[0])
            rows = []
            lat = []
            for k, feed in enumerate(samples):
                tt = time.perf_counter()
                outs = rknn_obj.inference(inputs=feed)
                lat.append((time.perf_counter() - tt) * 1000)
                lg = ac = None
                for o in outs:
                    a = np.asarray(o)
                    if a.ndim == 2 and a.shape[-1] == 2:
                        ac = a
                    else:
                        lg = a
                rows.append({"idx": k,
                             "logits": [float(v) for v in np.asarray(lg).reshape(-1)],
                             "act_logits": [float(v) for v in np.asarray(ac).reshape(-1)]})
                print("### sim[%02d] logits=%s act=%s"
                      % (k, rows[-1]["logits"], rows[-1]["act_logits"]), flush=True)
            ls = sorted(lat)
            print("### simulator latency ms: n=%d p50=%.2f min=%.2f max=%.2f"
                  % (len(ls), ls[len(ls) // 2], ls[0], ls[-1]), flush=True)
            with open(args.rank_dump, "w") as f:
                json.dump({"label": args.label, "onnx": ONNX, "target": args.target,
                           "quantized_dtype": args.quantized_dtype,
                           "do_quantization": args.do_quantization,
                           "input_names": IN_NAMES, "heldout": args.rank_heldout,
                           "rknn": OUT, "sim_latency_ms": ls[len(ls) // 2],
                           "samples": rows}, f, indent=2)
            print("### wrote", args.rank_dump, "elapsed=%.1fs" % (time.time() - t2), flush=True)
        except Exception as e:
            print("### RANK_DUMP_FAILED %s: %s" % (type(e).__name__, e), flush=True)
            traceback.print_exc()

    print("### out size bytes =", os.path.getsize(OUT), flush=True)
    print("### DONE_OK", flush=True)
except SystemExit:
    raise
except Exception:
    print("### EXCEPTION", flush=True)
    traceback.print_exc()
    sys.exit(20)
finally:
    try:
        rknn_obj.release()
    except Exception as e:
        print("### release warn:", e, flush=True)
    print("### total elapsed=%.1fs" % (time.time() - t0), flush=True)
