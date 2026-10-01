#!/usr/bin/env python3
"""Do the three minimal variants actually share weights? Compare every initializer pairwise.

If they differ, any cross-variant numeric difference (in ORT or on device) is a *different model*,
not a fusion effect, and the three-way comparison must be rebuilt on a matched triple.
"""
import hashlib
import json
import os

import numpy as np
import onnx
from onnx import numpy_helper

BASE = "/mnt/f/laya-sdpa-probe"
VARIANTS = {
    "noguard": f"{BASE}/onnx/var-noguard.s512.opset18.onnx",
    "guard": f"{BASE}/onnx/var-guard.s512.opset18.onnx",
    "sdpa": f"{BASE}/onnx/var-sdpa.s512.opset18.onnx",
}


def inits(path):
    m = onnx.load(path, load_external_data=True)
    d = {}
    for i in m.graph.initializer:
        a = numpy_helper.to_array(i)
        d[i.name] = a
    return d


def digest(a):
    return hashlib.md5(np.ascontiguousarray(a, dtype=np.float64).tobytes()).hexdigest()[:12]


def main():
    tabs = {k: inits(p) for k, p in VARIANTS.items()}
    for k, t in tabs.items():
        print(f"### {k}: {len(t)} initializers")
        for n, a in sorted(t.items()):
            print(f"###   {n:24s} shape={list(a.shape)} md5={digest(a)} sum={float(a.astype(np.float64).sum()):+.8e}")

    keys = sorted(set().union(*[set(t) for t in tabs.values()]))
    print("### pairwise:")
    for a in VARIANTS:
        for b in VARIANTS:
            if a >= b:
                continue
            same = diff = missing = 0
            for n in keys:
                x, y = tabs[a].get(n), tabs[b].get(n)
                if x is None or y is None:
                    missing += 1
                elif x.shape == y.shape and np.array_equal(x, y):
                    same += 1
                else:
                    diff += 1
            print(f"###   {a:8s} vs {b:8s} identical={same} different={diff} missing={missing}")

    # Also: is the 'mask' input actually consumed in each graph, or is it dead?
    import onnxruntime as ort
    rng = np.random.default_rng(7)
    hidden = rng.normal(0, 1, (1, 512, 768)).astype(np.float32)
    m0 = np.zeros((1, 12, 512, 512), dtype=np.float32)
    m1 = np.zeros((1, 12, 512, 512), dtype=np.float32)
    m1[..., -32:] = -10000.0
    for k, p in VARIANTS.items():
        s = ort.InferenceSession(p, providers=["CPUExecutionProvider"])
        names = [i.name for i in s.get_inputs()]
        o0 = s.run(None, {"hidden": hidden, "mask": m0})[0]
        o1 = s.run(None, {"hidden": hidden, "mask": m1})[0]
        d = np.abs(o0.astype(np.float64) - o1.astype(np.float64))
        print(f"### {k}: ort input names={names} mask_sensitivity max_abs_diff={d.max():.6e} "
              f"bitwise_equal={bool(np.array_equal(o0, o1))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
