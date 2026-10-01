"""ORT CPU run over the exported ONNX: golden tensors, torch-vs-ORT parity, onnxsim check."""
import hashlib
import json
import os

import numpy as np
import onnx
import onnxruntime as ort

ONNX_DIR = "/home/harve/laya-rknn/onnx"
GOLDEN = "/home/harve/laya-rknn/golden"
SRC = os.path.join(ONNX_DIR, "laya-multilingual.s512.m16.fp32.onnx")
SIM = os.path.join(ONNX_DIR, "laya-multilingual.s512.m16.fp32.sim.onnx")
DTYPES = {"tensor(float)": np.float32, "tensor(long)": np.int64, "tensor(bool)": np.bool_,
          "tensor(int64)": np.int64, "tensor(float16)": np.float16}


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def describe(sess, path):
    spec = {"inputs": [], "outputs": []}
    for kind, ios in (("inputs", sess.get_inputs()), ("outputs", sess.get_outputs())):
        for io in ios:
            spec[kind].append({"name": io.name, "dtype": io.type, "shape": list(io.shape)})
    return spec


def run(sess, feeds):
    outs = sess.run(None, feeds)
    names = [o.name for o in sess.get_outputs()]
    return dict(zip(names, outs))


def main():
    print("onnxruntime", ort.__version__)
    sess = ort.InferenceSession(SRC, providers=["CPUExecutionProvider"])
    print("providers:", sess.get_providers())

    spec = describe(sess, SRC)
    print("io spec:", json.dumps(spec, indent=2))

    feeds = {}
    print("--- input files ---")
    for io in spec["inputs"]:
        p = os.path.join(GOLDEN, io["name"] + ".npy")
        arr = np.load(p)
        feeds[io["name"]] = arr
        print("  %-15s file=%-22s numpy_dtype=%-9s shape=%s" % (io["name"], os.path.basename(p), arr.dtype, arr.shape))
        onnx_dtype = DTYPES.get(io["dtype"])
        if onnx_dtype is not None and arr.dtype != onnx_dtype:
            raise SystemExit("dtype mismatch for %s: npy=%s onnx=%s" % (io["name"], arr.dtype, io["dtype"]))

    out = run(sess, feeds)
    for name, arr in out.items():
        np.save(os.path.join(GOLDEN, name + ".npy"), arr)
        print("  ORT out %-12s dtype=%-9s shape=%s" % (name, arr.dtype, arr.shape))

    print("--- ORT golden logits[0,:8] ---")
    print(np.array2string(out["logits"][0, :8], precision=6, floatmode="maxprec"))
    print("--- ORT golden logits[0] full(16) ---")
    print(np.array2string(out["logits"][0], precision=6, floatmode="maxprec"))
    print("--- ORT golden act_logits ---")
    print(np.array2string(out["act_logits"][0], precision=6, floatmode="maxprec"))

    print("--- torch eager vs ORT ---")
    for name, ref in (("logits", "torchref_logits"), ("act_logits", "torchref_act_logits")):
        t = np.load(os.path.join(GOLDEN, ref + ".npy"))
        d = np.abs(t - out[name])
        print("  %-12s max_abs_diff=%.3e mean_abs_diff=%.3e" % (name, d.max(), d.mean()))

    # ---------------- onnxsim ----------------
    print("=== onnxsim ===")
    from onnxsim import simplify
    model = onnx.load(SRC)
    del model.graph.value_info[:]
    try:
        sim_model, ok = simplify(model, overwrite_input_shapes=None, perform_optimization=True)
    except Exception as e:
        print("SIMPLIFY FAILED (raw):", repr(e))
        ok = False
        sim_model = None
    print("simplify check =", ok)
    if ok:
        onnx.save(sim_model, SIM)
        sess2 = ort.InferenceSession(SIM, providers=["CPUExecutionProvider"])
        spec2 = describe(sess2, SIM)
        print("sim io spec:", json.dumps(spec2, indent=2))
        out2 = run(sess2, feeds)
        worst = 0.0
        for name in out:
            d = np.abs(out[name].astype(np.float64) - out2[name].astype(np.float64))
            print("  %-12s max_abs_diff=%.3e" % (name, d.max()))
            worst = max(worst, float(d.max()))
        print("SIM max_abs_diff across outputs = %.6e  (threshold 1e-3)" % worst)
        if worst > 1e-3:
            print("DECISION: diff > 1e-3 -> DISCARD sim version")
            os.remove(SIM)
            print("removed", SIM)
        else:
            print("DECISION: diff <= 1e-3 -> keep sim version")
            print("sim size=%d md5=%s" % (os.path.getsize(SIM), md5(SIM)))
    else:
        print("DECISION: simplify returned ok=False -> DISCARD sim version")

    print("=== artifacts ===")
    for p in (SRC, SIM):
        if os.path.exists(p):
            print("%s  size=%d  md5=%s" % (p, os.path.getsize(p), md5(p)))
        else:
            print("%s  (absent)" % p)
    print("DONE")


if __name__ == "__main__":
    main()
