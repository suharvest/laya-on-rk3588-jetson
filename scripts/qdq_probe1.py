#!/usr/bin/env python3
"""Control probe: is the embeddings LayerNorm output really sample-dependent?

Loads ONLY the fp32 graph, adds the two suspect tensors as outputs, and runs both
heldout rows. Also prints where the tensor's extreme sits (token position) so the
"max comes from a padding position" hypothesis can be checked directly.
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

for si in (0, 14):
    d = sorted(glob.glob(os.path.join(HELD, "sample_*")))[si]
    f = {}
    for i in sess.get_inputs():
        a = np.load(os.path.join(d, i.name + ".npy"))
        t = np.bool_ if "bool" in i.type else (np.int64 if "int64" in i.type else np.float32)
        f[i.name] = a.astype(t)
    out = sess.run(NAMES, f)
    print("===== sample %d  (%s)" % (si, os.path.basename(d)))
    print("   n_tokens(nonzero mask) = %d" % int(f["attention_mask"].sum()))
    for n, v in zip(NAMES, out):
        a = np.asarray(v, dtype=np.float64)
        flat = a.reshape(-1)
        i = int(np.argmax(np.abs(flat)))
        print("   %-52s shape=%s min=%.6f max=%.6f absmax=%.6f  argmax_flat=%d"
              % (n.split("/")[-1], a.shape, flat.min(), flat.max(), np.abs(flat).max(), i))
        if a.ndim == 3:
            t = i // a.shape[2]
            c = i % a.shape[2]
            print("        argmax at token %d channel %d ; first8=%s"
                  % (t, c, np.round(flat[:8], 4).tolist()))
    logits = np.asarray(out[2], dtype=np.float64).reshape(-1)
    print("   logits = %s" % np.round(logits[:8], 4).tolist())
