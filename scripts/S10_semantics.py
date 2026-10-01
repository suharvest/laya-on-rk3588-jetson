"""Do the two lineages actually carry the same attention mask, and in what value form?

Runs both fp32 graphs on one heldout row with the mask carriers promoted to graph outputs:
  oldship : the 22 /encoder/layers.N/attn/Where outputs
  newlin  : /encoder/Where_1_output_0 and /encoder/Where_2_output_0
plus, in both, every /encoder/layers.N/attn/Add_2 output (the tensor that the mask is added
into) and the graph logits.

Answers, from the numbers:
  - what the mask value sets are (0 / -inf / +-FLT_MAX / ...)
  - which layers each carrier feeds, and whether oldship's 8/14 split matches newlin's
  - whether Add_2 is bit-identical across lineages, i.e. whether the +-FLT_MAX form is
    numerically the same mask at the point where it is consumed
"""
import argparse
import collections
import glob
import os

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

ap = argparse.ArgumentParser()
ap.add_argument("--oldship", required=True)
ap.add_argument("--newlin", required=True)
ap.add_argument("--heldout", required=True)
ap.add_argument("--sample", default="sample_0000")
ap.add_argument("--out", default=None)
a = ap.parse_args()


def load(path):
    m = onnx.load(path, load_external_data=False)
    prod = {}
    for n in m.graph.node:
        for o in n.output:
            prod[o] = n
    cons = collections.defaultdict(list)
    for n in m.graph.node:
        for i in n.input:
            cons[i].append(n)
    return m, prod, cons


def expose(m, prod, names):
    have = {o.name for o in m.graph.output}
    added = []
    for n in names:
        if n in have or n not in prod:
            continue
        m.graph.output.append(helper.make_tensor_value_info(n, TensorProto.FLOAT, None))
        added.append(n)
    return added


def run(m, path, feeds, tag, extra, prod):
    added = expose(m, prod, extra)
    tmp = "/tmp/_s10_%s.onnx" % tag
    onnx.save(m, tmp)
    sess = ort.InferenceSession(tmp, providers=["CPUExecutionProvider"])
    inn = [i.name for i in sess.get_inputs()]
    outn = [o.name for o in sess.get_outputs()]
    print("### [%s] inputs=%s" % (tag, inn), flush=True)
    print("### [%s] exposed extra outputs=%d" % (tag, len(added)), flush=True)
    f = {k: feeds[k] for k in inn if k in feeds}
    missing = [k for k in inn if k not in feeds]
    if missing:
        raise SystemExit("### MISSING FEEDS for %s: %s (have %s)" % (tag, missing, sorted(feeds)))
    outs = sess.run(outn, f)
    del sess
    os.remove(tmp)
    return dict(zip(outn, outs))


def desc(x):
    x = np.asarray(x)
    fl = x.reshape(-1)
    vals, cnts = np.unique(fl, return_counts=True)
    order = np.argsort(-cnts)
    top = [(float(vals[i]), int(cnts[i])) for i in order[:6]]
    return "shape=%s n_unique=%d top=[%s]" % (tuple(x.shape), len(vals),
                                              ", ".join("%r x%d" % t for t in top))


d = os.path.join(a.heldout, a.sample)
feeds = {os.path.basename(p)[:-4]: np.load(p) for p in sorted(glob.glob(os.path.join(d, "*.npy")))}
print("### feeds: %s" % {k: (v.shape, str(v.dtype)) for k, v in feeds.items()}, flush=True)

mo, po, co = load(a.oldship)
mn, pn, cn = load(a.newlin)

layers = sorted({int(n.name.split("/encoder/layers.")[1].split("/")[0])
                 for n in mo.graph.node if n.name.startswith("/encoder/layers.")})
print("### layers in oldship: n=%d %s" % (len(layers), layers), flush=True)
layers_n = sorted({int(n.name.split("/encoder/layers.")[1].split("/")[0])
                   for n in mn.graph.node if n.name.startswith("/encoder/layers.")})
print("### layers in newlin : n=%d %s" % (len(layers_n), layers_n), flush=True)

o_where = [n.output[0] for n in mo.graph.node
           if n.op_type == "Where" and n.name.startswith("/encoder/layers.")]
o_add2 = ["/encoder/layers.%d/attn/Add_2" % L for L in layers]
n_carriers = ["/encoder/Where_1_output_0", "/encoder/Where_2_output_0"]
n_add2 = ["/encoder/layers.%d/attn/Add_2" % L for L in layers_n]

print("### oldship per-layer Where (%d): %s" % (len(o_where), o_where), flush=True)
for t in n_carriers:
    print("### newlin carrier %-32s producers: %s consumers=%d"
          % (t, pn[t].op_type if t in pn else "ABSENT", len(cn[t])), flush=True)

ro = run(mo, a.oldship, feeds, "old", o_where + o_add2, po)
rn = run(mn, a.newlin, feeds, "new", n_carriers + n_add2, pn)

print("\n#### oldship per-layer Where masks", flush=True)
for t in o_where:
    print("   %-46s %s" % (t, desc(ro[t])), flush=True)

print("\n#### newlin carriers", flush=True)
for t in n_carriers:
    print("   %-46s %s" % (t, desc(rn[t])), flush=True)

print("\n#### which layer consumes which carrier", flush=True)
layer2carrier = {}
for t in n_carriers:
    for c in cn[t]:
        if c.name.startswith("/encoder/layers.") and c.op_type == "Add":
            L = int(c.name.split("/encoder/layers.")[1].split("/")[0])
            layer2carrier[L] = t
            print("   layer %2d <- %s" % (L, t), flush=True)

print("\n#### mask comparison oldship[layer] vs newlin[layer]", flush=True)
newlin_by_layer = {}
for t, Ls in (("/encoder/Where_1_output_0", None), ("/encoder/Where_2_output_0", None)):
    for L, tt in layer2carrier.items():
        if tt == t:
            newlin_by_layer[L] = rn[t]

for t in o_where:
    L = int(t.split("/encoder/layers.")[1].split("/")[0])
    mo_ = np.asarray(ro[t], np.float32)
    if L in newlin_by_layer:
        mn_ = np.asarray(newlin_by_layer[L], np.float32)
        same = mo_.tobytes() == mn_.tobytes()
        fin_o = np.isfinite(mo_)
        print("   layer %2d old=%-32s new=%-32s bitwise=%s  old[nonzero]%s new[nonzero]%s"
              % (L, desc(mo_)[:28], desc(mn_)[:28], same,
                 desc(mo_[fin_o & (mo_ != 0)]) if np.any(fin_o & (mo_ != 0)) else "{}",
                 desc(mn_[mn_ != 0])[:40] if np.any(mn_ != 0) else "{}"), flush=True)
    else:
        print("   layer %2d old=%-32s newlin=NO CARRIER" % (L, desc(mo_)[:28]), flush=True)

print("\n#### Add_2 (masked scores) bitwise oldship vs newlin", flush=True)
nbit = 0
for L in layers:
    t = "/encoder/layers.%d/attn/Add_2" % L
    if t not in ro or t not in rn:
        print("   layer %2d Add_2 missing old=%s new=%s" % (L, t in ro, t in rn), flush=True)
        continue
    A = np.asarray(ro[t], np.float32)
    B = np.asarray(rn[t], np.float32)
    eq = A.tobytes() == B.tobytes()
    nbit += int(eq)
    print("   layer %2d Add_2 shape=%s bitwise=%s maxabs=%.6g %s"
          % (L, tuple(A.shape), eq, float(np.max(np.abs(A.astype(np.float64) - B.astype(np.float64)))),
             desc(A)[:56]), flush=True)
print("### Add_2 bit-exact %d/%d" % (nbit, len(layers)), flush=True)

lg_o = np.asarray(ro["logits"], np.float32)
lg_n = np.asarray(rn["logits"], np.float32)
print("### logits bitwise=%s maxabs=%.6g" % (lg_o.tobytes() == lg_n.tobytes(),
                                             float(np.max(np.abs(lg_o - lg_n)))), flush=True)
print("### S10_DONE", flush=True)
