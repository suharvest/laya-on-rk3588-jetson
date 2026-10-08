#!/usr/bin/env python3
"""Control probe 3: walk the embedding path raw for rows 0 and 14.

Gather output -> LayerNorm output, plus the raw input_ids, in one session, so the
"identical across rows" observation can be pinned to a specific stage.
"""
import glob
import os

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper

REF = "/home/harvest/laya-trt/onnx/laya-multilingual.s512.m16.fp32.ng.onnx"
HELD = "/home/harvest/laya-trt/calib_s512"
NAMES = ["/encoder/embeddings/tok_embeddings/Gather_output_0",
         "/encoder/embeddings/norm/LayerNormalization_output_0",
         "/encoder/layers.0/attn/Wqkv/MatMul_output_0",
         "logits"]

so = ort.SessionOptions()
so.intra_op_num_threads = 6
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
so.log_severity_level = 3

m = onnx.load(REF)
have = {o.name for o in m.graph.output}
for n in NAMES:
    if n not in have:
        vi = helper.ValueInfoProto()
        vi.name = n
        vi.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
        m.graph.output.append(vi)
blob = m.SerializeToString()
del m
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
    ids = f["input_ids"].reshape(-1)
    print("===== sample %d  ids[0:8]=%s  ids[10:14]=%s  nz=%d"
          % (si, ids[:8].tolist(), ids[10:14].tolist(), int((ids != 0).sum())))
    print("      marker_pos=%s qtype=%s" % (f["marker_pos"].reshape(-1)[:4].tolist(), f["qtype"].reshape(-1).tolist()))
    for n, v in zip(NAMES, outs[si]):
        a = np.asarray(v, dtype=np.float64)
        print("      %-34s first8=%s  min=%.6f max=%.6f"
              % (n.split("/")[-1][:34], np.round(a.reshape(-1)[:8], 6).tolist(),
                 a.min(), a.max()))

print("\n===== per-tensor equality rows 0 vs 14 =====")
for n, a, b in zip(NAMES, outs[0], outs[14]):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    d = np.abs(a - b)
    print("  %-34s equal=%s maxdiff=%.8f ndiff=%d"
          % (n.split("/")[-1][:34], np.array_equal(a, b), d.max(), int((d > 0).sum())))

print("\n===== do the weights differ? raw Gather rows for the two id vectors =====")
wi = [i for i in onnx.load(REF).graph.initializer if "tok_embeddings" in i.name]
print("  tok_embeddings initializers:", [(i.name, list(i.dims)) for i in wi][:4])
