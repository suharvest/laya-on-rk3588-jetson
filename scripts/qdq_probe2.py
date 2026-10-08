#!/usr/bin/env python3
"""Control probe 2: are the two rows' encoder inputs really the same tensor?

Prints whether the embeddings-layer tensors are array-equal between rows 0 and 14,
and where input_ids / attention_mask / marker_pos enter the graph.
"""
import glob
import os

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper

REF = "/home/harvest/laya-trt/onnx/laya-multilingual.s512.m16.fp32.ng.onnx"
HELD = "/home/harvest/laya-trt/calib_s512"
NAMES = ["/encoder/embeddings/norm/LayerNormalization_output_0",
         "/encoder/layers.0/attn/Wqkv/MatMul_output_0",
         "/encoder/layers.0/input/LayerNormalization_output_0",
         "logits"]

print("===== who consumes the graph inputs =====")
m0 = onnx.load(REF)
for want in ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"):
    cons = [(n.op_type, n.name) for n in m0.graph.node if want in list(n.input)]
    print("  %-16s <- %s" % (want, cons[:4]))
prods = {o: n for n in m0.graph.node for o in n.output}
for probe in ("/encoder/embeddings/norm/LayerNormalization_output_0",
              "/encoder/layers.0/attn/Wqkv/MatMul_output_0",
              "/encoder/layers.0/input/LayerNormalization_output_0"):
    n = prods.get(probe)
    print("  producer of %s -> %s" % (probe, (n.op_type, list(n.input)) if n else None))
have = {o.name for o in m0.graph.output}
for n in NAMES:
    if n not in have:
        vi = helper.ValueInfoProto()
        vi.name = n
        vi.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
        m0.graph.output.append(vi)
blob = m0.SerializeToString()
del m0
so = ort.SessionOptions()
so.intra_op_num_threads = 6
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
so.log_severity_level = 3
sess = ort.InferenceSession(blob, so, providers=["CPUExecutionProvider"])
del blob

outs = {}
for si in (0, 14):
    d = sorted(glob.glob(os.path.join(HELD, "sample_*")))[si]
    f = {}
    for i in sess.get_inputs():
        a = np.load(os.path.join(d, i.name + ".npy"))
        t = np.bool_ if "bool" in i.type else (np.int64 if "int64" in i.type else np.float32)
        f[i.name] = a.astype(t)
    outs[si] = sess.run(NAMES, f)
    print("===== sample %d ids[0:6]=%s mask_nz=%d" %
          (si, f["input_ids"].reshape(-1)[:6].tolist(), int(f["attention_mask"].sum())))

print("\n===== equality between rows 0 and 14 =====")
for n, a, b in zip(NAMES, outs[0], outs[14]):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    same = np.array_equal(a, b)
    print("  %-52s equal=%s  maxdiff=%.8f" % (n.split("/")[-1], same, np.abs(a - b).max()))
    if not same and a.ndim == 3:
        diff = np.abs(a - b).reshape(-1)
        nz = np.nonzero(diff > 0)[0]
        print("      differing elements: %d / %d ; first at flat %d" % (len(nz), diff.size, nz[0] if len(nz) else -1))
        tok = nz // a.shape[2]
        print("      differing tokens (unique, first 20): %s" % np.unique(tok)[:20].tolist())
