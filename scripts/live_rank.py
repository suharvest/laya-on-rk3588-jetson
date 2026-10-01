"""Live-class live-order comparison, same protocol as scripts/46_rank_from_dumps.py.

Primary number = per-sample order of the LIVE (masked-in) markers; rel_l2 is computed on the live
slice only, because the -10000 mask sentinels otherwise dominate the norm.
"""
import argparse
import collections
import glob
import json
import os

import numpy as np


def kendall(a, b):
    n = len(a)
    if n < 2:
        return 1.0
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
            if s > 0:
                conc += 1
            elif s < 0:
                disc += 1
    return (conc - disc) / (n * (n - 1) / 2.0)


def load_dump(path):
    with open(path) as f:
        j = json.load(f)
    return {int(s["idx"]): s for s in j["samples"]}


ap = argparse.ArgumentParser()
ap.add_argument("--ref", required=True, help="ORT fp32 reference dump json")
ap.add_argument("--dump", action="append", default=[], help="name=json")
ap.add_argument("--heldout", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

masks = []
for d in sorted(glob.glob(os.path.join(args.heldout, "sample_*"))):
    masks.append(np.load(os.path.join(d, "marker_mask.npy")).reshape(-1).astype(bool))
print("### heldout=%s samples=%d live_sizes=%s"
      % (args.heldout, len(masks), collections.Counter(int(m.sum()) for m in masks)), flush=True)

ref = load_dump(args.ref)
print("### ref %s samples=%d" % (args.ref, len(ref)), flush=True)

result = {"ref": args.ref, "heldout": args.heldout, "pairs": {}}
for kv in args.dump:
    name, path = kv.split("=", 1)
    d = load_dump(path)
    rows = []
    for idx in sorted(ref):
        if idx not in d:
            continue
        live = masks[idx] if idx < len(masks) else np.ones(len(ref[idx]["logits"]), bool)
        x = np.asarray(ref[idx]["logits"], float)[live]
        y = np.asarray(d[idx]["logits"], float)[live]
        ro, go = np.argsort(-x, kind="stable"), np.argsort(-y, kind="stable")
        rows.append({"idx": idx, "n_live": int(len(x)),
                     "top1_match": bool(int(ro[0]) == int(go[0])),
                     "order_exact": bool(np.array_equal(ro, go)),
                     "kendall_tau": round(float(kendall(x, y)), 6),
                     "rel_l2": round(float(np.linalg.norm(x - y) / max(1e-12, np.linalg.norm(x))), 6),
                     "max_abs_diff": round(float(np.max(np.abs(x - y))), 6),
                     "ref_order": [int(v) for v in ro], "trt_order": [int(v) for v in go],
                     "ref_vals": [round(float(v), 4) for v in x],
                     "trt_vals": [round(float(v), 4) for v in y]})
    n = len(rows)
    summ = {"n": n,
            "top1_match": "%d/%d" % (sum(r["top1_match"] for r in rows), n),
            "order_exact": "%d/%d" % (sum(r["order_exact"] for r in rows), n),
            "kendall_tau_min": round(float(np.min([r["kendall_tau"] for r in rows])), 6),
            "rel_l2_mean": round(float(np.mean([r["rel_l2"] for r in rows])), 6),
            "rel_l2_max": round(float(np.max([r["rel_l2"] for r in rows])), 6),
            "max_abs_diff_max": round(float(np.max([r["max_abs_diff"] for r in rows])), 6)}
    summ["verdict"] = ("PASS" if all(r["order_exact"] for r in rows)
                       else ("PARTIAL" if all(r["top1_match"] for r in rows) else "FAIL"))
    result["pairs"][name] = {"summary": summ, "rows": rows}
    print("### %-28s top1=%s exact=%s tau_min=%.4f rel_l2_mean=%.6f rel_l2_max=%.6f maxabs=%.5f -> %s"
          % (name, summ["top1_match"], summ["order_exact"], summ["kendall_tau_min"],
             summ["rel_l2_mean"], summ["rel_l2_max"], summ["max_abs_diff_max"], summ["verdict"]),
          flush=True)
    print("    idx live top1 exact  tau      rel_l2    ref_order -> trt_order", flush=True)
    for r in rows:
        print("    %2d  %2d   %-5s %-5s %.4f  %.6f  %s -> %s%s"
              % (r["idx"], r["n_live"], r["top1_match"], r["order_exact"], r["kendall_tau"],
                 r["rel_l2"], r["ref_order"], r["trt_order"], "" if r["order_exact"] else "   DIFF"),
              flush=True)

with open(args.out, "w") as f:
    json.dump(result, f, indent=2)
print("### wrote", args.out, flush=True)
