#!/usr/bin/env python3
"""Inspect the two minimal SDPA ONNX variants, build deterministic inputs, and produce ORT fp32 refs.

Outputs (all new files, nothing existing is touched):
  /mnt/f/laya-sdpa-probe/v_three/inputs/<set>/<name>.npy      -- dtypes taken from the ONNX itself
  /mnt/f/laya-sdpa-probe/v_three/ort/<variant>/<set>/<out>.npy
  /mnt/f/laya-sdpa-probe/v_three/v1_report.json
"""
import json
import os

import numpy as np
import onnx
import onnxruntime as ort

BASE = "/mnt/f/laya-sdpa-probe"
VARIANTS = {
    "noguard": f"{BASE}/onnx/var-noguard.s512.opset18.onnx",
    "guard": f"{BASE}/onnx/var-guard.s512.opset18.onnx",
}
OUT = f"{BASE}/v_three"
SEQ = 512
HEADS = 12
HID = 768
PAD_KEYS = 32          # last N key positions masked in the "maskpad" set
MASK_MIN = -10000.0    # same finite sentinel the laya lineage uses


def tinfo(t):
    tt = t.type.tensor_type
    dims = []
    for d in tt.shape.dim:
        if d.HasField("dim_value"):
            dims.append(int(d.dim_value))
        elif d.HasField("dim_param"):
            dims.append(d.dim_param)
        else:
            dims.append(None)
    return {"name": t.name,
            "elem_type": onnx.TensorProto.DataType.Name(tt.elem_type),
            "dims": dims}


def describe(path):
    m = onnx.load(path)
    g = m.graph
    d = {"path": path, "size_bytes": os.path.getsize(path),
         "ir_version": m.ir_version,
         "opset_import": [(o.domain or "ai.onnx", int(o.version)) for o in m.opset_import],
         "producer": f"{m.producer_name} {m.producer_version}".strip(),
         "inputs": [tinfo(t) for t in g.input],
         "outputs": [tinfo(t) for t in g.output],
         "n_nodes": len(g.node)}
    hist = {}
    for n in g.node:
        hist[n.op_type] = hist.get(n.op_type, 0) + 1
    d["op_hist"] = hist
    d["node_order"] = [f"{n.op_type}:{n.name}" for n in g.node]
    # multi-dim initializers = baked constants worth knowing about (mask constants live here)
    big = []
    for init in g.initializer:
        nd = len(init.dims)
        if nd >= 3:
            arr = onnx.numpy_helper.to_array(init)
            big.append({"name": init.name, "dims": [int(x) for x in init.dims],
                        "dtype": str(arr.dtype),
                        "min": float(arr.min()), "max": float(arr.max()),
                        "n_unique": int(np.unique(arr).size),
                        "n_masked_min": int((arr <= -9999).sum())})
    d["big_initializers"] = big
    return d, m


def build_inputs(set_name, spec):
    """Deterministic inputs keyed off the ONNX's own declared names/shapes/dtypes."""
    rng = np.random.default_rng(20260929)
    out = {}
    for item in spec["inputs"]:
        name, et, dims = item["name"], item["elem_type"], item["dims"]
        shape = [SEQ if d in (None, -1, "seq") else int(d) for d in dims]
        if name == "hidden":
            arr = rng.normal(0.0, 1.0, size=shape).astype(np.float32)
        elif name == "mask":
            arr = np.zeros(shape, dtype=np.float32)
            if set_name == "maskpad":
                arr[..., SEQ - PAD_KEYS:] = MASK_MIN
        else:
            raise SystemExit(f"unexpected model input {name!r} {dims} {et}")
        np_dt = {"FLOAT": np.float32, "FLOAT16": np.float16, "DOUBLE": np.float64,
                 "INT64": np.int64, "INT32": np.int32, "BOOL": np.bool_}[et]
        arr = arr.astype(np_dt)
        d = os.path.join(OUT, "inputs", set_name)
        os.makedirs(d, exist_ok=True)
        np.save(os.path.join(d, name + ".npy"), arr)
        out[name] = arr
    return out


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
            "rel_l2": float(np.linalg.norm(d) / nb) if nb else None}


def main():
    os.makedirs(OUT, exist_ok=True)
    report = {"variants": {}, "input_sets": ["mask0", "maskpad"], "ort": {}, "pairs": {}}

    specs, models = {}, {}
    for k, p in VARIANTS.items():
        d, m = describe(p)
        specs[k], models[k] = d, m
        report["variants"][k] = d
        print(f"### variant {k}")
        print(f"    size={d['size_bytes']} opset={d['opset_import']} nodes={d['n_nodes']}")
        print(f"    inputs={json.dumps(d['inputs'])}")
        print(f"    outputs={json.dumps(d['outputs'])}")
        print(f"    op_hist={json.dumps(d['op_hist'])}")
        print(f"    big_initializers={json.dumps(d['big_initializers'])}")

    sessions = {k: ort.InferenceSession(VARIANTS[k], providers=["CPUExecutionProvider"])
                for k in VARIANTS}

    for set_name in report["input_sets"]:
        for k in VARIANTS:
            feed = build_inputs(set_name, specs[k])
            outs = sessions[k].run(None, feed)
            names = [o.name for o in models[k].graph.output]
            dd = os.path.join(OUT, "ort", k, set_name)
            os.makedirs(dd, exist_ok=True)
            entry = {}
            for nm, o in zip(names, outs):
                np.save(os.path.join(dd, nm + ".npy"), np.asarray(o))
                entry[nm] = stats(o)
            report["ort"].setdefault(k, {})[set_name] = entry
            print(f"### ORT {k} [{set_name}] {json.dumps(entry)}")

    # ORT(noguard) vs ORT(guard): are the two graphs numerically the same?
    for set_name in report["input_sets"]:
        a = np.load(os.path.join(OUT, "ort", "noguard", set_name, "out.npy"))
        b = np.load(os.path.join(OUT, "ort", "guard", set_name, "out.npy"))
        c = cmp(a, b)
        report["pairs"][f"ort_noguard_vs_ort_guard__{set_name}"] = c
        print(f"### ORT(noguard) vs ORT(guard) [{set_name}] {json.dumps(c)}")

    with open(os.path.join(OUT, "v1_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("### WROTE", os.path.join(OUT, "v1_report.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
