"""Equivalence check for the fltmax-value-form rewrite, plus the cross-lineage support check.

Required by the task: the rewritten graph must be bit-exact against the unmodified shared-mask
graph on logits / act_logits over all heldout rows. The mask tensors themselves cannot be
bit-exact by construction (the sentinel value is what changes), so for the masks this compares
the BLOCKED POSITION SET instead, and additionally compares that set against the blocked set of
the newlin carriers, which is the correspondence the whole experiment rests on.

Exit 0 only when logits and act_logits are bit-exact on every row and the blocked sets agree.
"""
import argparse
import collections
import glob
import json
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

ap = argparse.ArgumentParser()
ap.add_argument("--shared", required=True, help="unmodified shared-mask onnx (baseline)")
ap.add_argument("--flt", required=True, help="fltmax-value-form onnx (candidate)")
ap.add_argument("--newlin", default=None, help="newlin onnx, for the support cross-check")
ap.add_argument("--heldout", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--scratch", default="/mnt/f/laya-oldship/structprobe/scratch")
args = ap.parse_args()
os.makedirs(args.scratch, exist_ok=True)


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
    added = [n for n in names if n in prod and n not in have]
    for n in added:
        dt = TensorProto.BOOL if prod[n].op_type in ("Not", "And", "Or", "Equal") else TensorProto.FLOAT
        m.graph.output.append(helper.make_tensor_value_info(n, dt, None))
    return added


def layer_map(cons, carrier_names):
    out = {}
    for t in carrier_names:
        for c in cons[t]:
            if c.name.startswith("/encoder/layers.") and c.op_type == "Add":
                L = int(c.name.split("/encoder/layers.")[1].split("/")[0])
                out[L] = t
    return out


def run(path, tag, heldout, expose_names, limit, scratch):
    m, prod, cons = load(path)
    added = expose(m, prod, expose_names)
    tmp = os.path.join(scratch, "_s12_%s.onnx" % tag)
    onnx.save(m, tmp)
    print("### [%s] exposed=%s" % (tag, added), flush=True)
    sess = ort.InferenceSession(tmp, providers=["CPUExecutionProvider"])
    inn = [i.name for i in sess.get_inputs()]
    outn = [o.name for o in sess.get_outputs()]
    dirs = sorted(glob.glob(os.path.join(heldout, "sample_*")))
    if limit:
        dirs = dirs[:limit]
    rows = []
    for d in dirs:
        feeds = {n: np.load(os.path.join(d, n + ".npy")) for n in inn}
        rows.append(dict(zip(outn, sess.run(outn, feeds))))
    del sess
    os.remove(tmp)
    return rows, outn, cons


shared_carriers = ["/encoder/SharedMask_0_output_0", "/encoder/SharedMask_1_output_0"]
flt_diag = shared_carriers + ["/encoder/SharedMaskF_cast_output_0", "/encoder/SharedMaskF_sub_output_0",
                              "/encoder/SharedMaskF_not_output_0"]
newlin_carriers = ["/encoder/Where_1_output_0", "/encoder/Where_2_output_0"]

rs, osn, cs = run(args.shared, "shared", args.heldout, shared_carriers, args.limit, args.scratch)
rf, ofn, cf = run(args.flt, "flt", args.heldout, flt_diag, args.limit, args.scratch)
rn = on = cn = None
if args.newlin:
    rn, on, cn = run(args.newlin, "newlin", args.heldout, newlin_carriers, args.limit, args.scratch)

lmap_s = layer_map(cs, shared_carriers)
lmap_f = layer_map(cf, flt_diag)
lmap_n = layer_map(cn, newlin_carriers) if cn else {}
print("### layer->carrier  shared=%s" % dict(sorted(lmap_s.items())), flush=True)
print("### layer->carrier  flt   =%s" % dict(sorted(lmap_f.items())), flush=True)
if cn:
    print("### layer->carrier  newlin=%s" % dict(sorted(lmap_n.items())), flush=True)
    if lmap_n != lmap_s:
        print("### WARN: newlin layer->carrier map differs from the shared graph's", flush=True)

summary = {"rows": len(rs), "logits_bit_exact": 0, "act_bit_exact": 0,
           "shared_blocked": {}, "flt_blocked": {}, "newlin_blocked": {},
           "per_row": []}


def blocked(x):
    x = np.asarray(x, np.float32)
    return ~np.isfinite(x) | (np.abs(x) > 1.0e30)


def vset(x):
    x = np.asarray(x, np.float32)
    v, c = np.unique(x.reshape(-1), return_counts=True)
    return sorted(zip([float(a) for a in v], [int(b) for b in c]))


print("\n#### mask value sets (row 0)", flush=True)
for t in shared_carriers:
    print("   shared %-32s %s" % (t, vset(rs[0][t])), flush=True)
for t in shared_carriers:
    print("   flt    %-32s %s" % (t, vset(rf[0][t])), flush=True)
if rn:
    for t in newlin_carriers:
        print("   newlin %-32s %s" % (t, vset(rn[0][t])), flush=True)

print("\n#### blocked-position sets, shared vs flt vs newlin (row 0)", flush=True)
support = {}
for L in sorted(lmap_s):
    ts = lmap_s[L]
    tf = lmap_f.get(L, ts)
    bs = blocked(rs[0][ts])
    bf = blocked(rf[0][tf]) if tf in rf[0] else None
    line = "   layer %2d shared=%-30s n_blocked=%-5d flt=%-30s n_blocked=%-5d identical=%s" % (
        L, ts.split("/")[-1], int(bs.sum()), tf.split("/")[-1],
        int(bf.sum()) if bf is not None else -1,
        bool(bf is not None and np.array_equal(bs, bf)))
    if rn is not None and L in lmap_n:
        tn = lmap_n[L]
        bn = blocked(rn[0][tn])
        line += "  newlin=%-30s n_blocked=%-5d identical=%s" % (
            tn.split("/")[-1], int(bn.sum()), bool(np.array_equal(bn, bf if bf is not None else bs)))
        summary["newlin_blocked"][str(L)] = int(bn.sum())
    print(line, flush=True)
    summary["shared_blocked"][str(L)] = int(bs.sum())
    summary["flt_blocked"][str(L)] = int(bf.sum()) if bf is not None else None
    support[L] = (bs, bf)

print("\n#### per-row bitwise logits / act_logits (shared vs flt)", flush=True)
bad = 0
for k in range(len(rs)):
    lg_s = np.asarray(rs[k]["logits"], np.float32)
    lg_f = np.asarray(rf[k]["logits"], np.float32)
    ac_s = np.asarray(rs[k]["act_logits"], np.float32)
    ac_f = np.asarray(rf[k]["act_logits"], np.float32)
    e_lg = lg_s.tobytes() == lg_f.tobytes()
    e_ac = ac_s.tobytes() == ac_f.tobytes()
    summary["logits_bit_exact"] += int(e_lg)
    summary["act_bit_exact"] += int(e_ac)
    if not (e_lg and e_ac):
        bad += 1
    print("   row %02d logits_bit_exact=%-5s maxabs=%.6g  act_bit_exact=%-5s maxabs=%.6g"
          % (k, e_lg, float(np.max(np.abs(lg_s - lg_f))), e_ac, float(np.max(np.abs(ac_s - ac_f)))),
          flush=True)
    if rn is not None:
        lg_n = np.asarray(rn[k]["logits"], np.float32)
        print("        (shared vs newlin logits bit_exact=%s)" % (lg_s.tobytes() == lg_n.tobytes()),
              flush=True)
    summary["per_row"].append({"row": k, "logits_bit_exact": bool(e_lg), "act_bit_exact": bool(e_ac)})

mask_ok = all(bf is not None and np.array_equal(bs, bf) for bs, bf in support.values())
nl_ok = None
if rn is not None:
    nl_ok = all(np.array_equal(support[L][1], blocked(rn[0][lmap_n[L]])) for L in support if L in lmap_n)

summary["verdict"] = ("BIT_EXACT" if bad == 0 and mask_ok else
                      ("MASK_SUPPORT_MISMATCH" if bad == 0 else "DIVERGENT"))
summary["logits_act_ok"] = bad == 0
summary["blocked_sets_shared_eq_flt"] = bool(mask_ok)
summary["blocked_sets_shared_eq_newlin"] = nl_ok
print("\n### SUMMARY rows=%d logits_bit_exact=%d/%d act_bit_exact=%d/%d blocked_sets_shared_eq_flt=%s "
      "blocked_sets_shared_eq_newlin=%s -> %s"
      % (len(rs), summary["logits_bit_exact"], len(rs), summary["act_bit_exact"], len(rs),
         mask_ok, nl_ok, summary["verdict"]), flush=True)
with open(args.out, "w") as f:
    json.dump(summary, f, indent=2)
print("### wrote %s" % args.out, flush=True)
print("### S12_DONE", flush=True)
sys.exit(0 if summary["verdict"] == "BIT_EXACT" else 1)
