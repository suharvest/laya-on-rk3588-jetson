#!/usr/bin/env python3
"""Falsification test: is the unfused model's mask input truly inert, or is its output stuck?

Varying `hidden` must move the output (else the model is dead, which is a different defect).
Non-uniform masks must move a correct model's output; uniform masks must not (softmax shift
invariance), so only non-uniform masks are used here.
"""
import sys

import numpy as np
from rknnlite.api import RKNNLite

sys.path.insert(0, "/home/radxa/v_three")
from probe_laya_rknn import RKNN_TYPE_MAP  # noqa: E402

ROOT = "/home/radxa/v_three/"
H = np.load(ROOT + "inputs/mask0/hidden.npy").astype(np.float16)
M = np.transpose(np.load(ROOT + "inputs/mask0/mask.npy"), (0, 2, 3, 1)).astype(np.float16)

rng = np.random.default_rng(5)
# NHWC layout: M[0, q, k, c]; mask the first half of the KEY axis, non-uniform across q/k.
m_firsthalf = np.zeros(M.shape, dtype=np.float16)
m_firsthalf[:, :, :256, :] = -10000.0
variants = {
    "hidden_x2": (H * 2.0, M),
    "mask_firsthalf": (H, m_firsthalf),
    "mask_randx5": (H, (rng.normal(0, 1, M.shape) * 5.0).astype(np.float16)),
}

for lab, path in (("noguard", "models/var-noguard.rk3588.rknn"),
                  ("guard", "models/var-guard.rk3588.rknn")):
    r = RKNNLite(verbose=False)
    if r.load_rknn(ROOT + path) != 0 or r.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2) != 0:
        print("### %s LOAD/INIT FAILED" % lab)
        continue
    base = np.asarray(r.inference(inputs=[H, M])[0])
    for name, (h, m) in variants.items():
        hh = h.astype(np.float16)
        mm = m.astype(np.float16)
        o = np.asarray(r.inference(inputs=[hh, mm])[0])
        d = np.abs(o.astype(np.float64) - base.astype(np.float64))
        print("### %-8s vary %-16s bitwise_equal=%-5s max_abs_diff=%.6e base_absmax=%.6f new_absmax=%.6f"
              % (lab, name, bool(np.array_equal(o, base)), d.max(),
                 float(np.abs(base).max()), float(np.abs(o).max())), flush=True)
    r.release()
print("### SENS_DONE")
