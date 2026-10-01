#!/usr/bin/env python3
"""Per-row live-class ordering: explicit-Q/DQ INT8 vs TRT fp16 vs ORT fp32 (s512, 16 heldout rows).

Live entries = marker_mask == True. The masked positions carry the -10000 sentinel and would
dominate any norm, so rel_l2 is computed on live entries only (same rule as int8_compare.py).
"""
import glob, json, os, sys
import numpy as np

BASE = "/home/harvest/laya-trt"
HELD = os.path.join(BASE, "calib_s512")
ORT = os.path.join(BASE, "int8/ortdump.s512_ng.json")

VARIANTS = [
    ("fp16",      "qdq/out/rank_fp16_nx_graph.json"),
    ("fp16_nogr", "qdq/out/rank_fp16_nx_nograph.json"),
    ("qdq_nx",    "qdq/out/rank_qdq_nx_graph.json"),
    ("qdq_nx_nogr", "qdq/out/rank_qdq_nx_nograph.json"),
    ("qdq_nano",  "qdq/out/rank_qdq_nano_graph.json"),
    ("qdq_nano_nogr", "qdq/out/rank_qdq_nano_nograph.json"),
]

def load(p):
    with open(p) as f:
        return json.load(f)

def live(logits, mask):
    return np.asarray(logits, dtype=np.float64).reshape(-1)[mask]

def order(x):
    return list(int(v) for v in np.argsort(-x, kind="stable"))

def rel_l2(a, b):
    d = np.linalg.norm(a - b); n = np.linalg.norm(b)
    return float(d / n) if n > 0 else float("nan")

def main():
    dirs = sorted(glob.glob(os.path.join(HELD, "sample_*")))
    masks = [np.load(os.path.join(d, "marker_mask.npy")).reshape(-1).astype(bool) for d in dirs]
    ref = load(ORT)
    V = {}
    for tag, p in VARIANTS:
        fp = os.path.join(BASE, p)
        if os.path.exists(fp):
            V[tag] = load(fp)
        else:
            print("### missing %s" % fp)
    print("### ORT ref samples=%d  variants=%s" % (len(ref["samples"]), list(V)))

    # global row identity: use ORT row order
    ref_live = [live(s["logits"], m) for s, m in zip(ref["samples"], masks)]
    ref_ord = [order(x) for x in ref_live]

    summary = {}
    for tag, d in V.items():
        smp = d["samples"]
        rows = []
        for i, (s, m) in enumerate(zip(smp, masks)):
            lv = live(s["logits"], m)
            o = order(lv)
            exact_vs_ort = (o == ref_ord[i])
            r = rel_l2(lv, ref_live[i]) if len(lv) == len(ref_live[i]) else float("nan")
            rows.append({"idx": s["idx"], "live": int(m.sum()), "order": o,
                         "exact_vs_ort": bool(exact_vs_ort), "rel_l2": r})
        summary[tag] = rows

    print("\n===== PER-ROW LIVE ORDER (16 heldout) =====")
    hdr = "row  nl  ORT" + "".join("%22s" % t for t in summary)
    print(hdr)
    for i in range(len(ref["samples"])):
        line = "%3d %3d  %-18s" % (i, int(masks[i].sum()), ref_ord[i])
        for t in summary:
            r = summary[t][i]
            mark = "=" if r["exact_vs_ort"] else "X"
            line += "  %-16s %s" % (r["order"], mark)
        print(line)

    print("\n===== rel_l2 vs ORT fp32 (live entries only) =====")
    print("row  " + "".join("%14s" % t for t in summary))
    for i in range(len(ref["samples"])):
        print("%3d  " % i + "".join("%14.5f" % summary[t][i]["rel_l2"] for t in summary))

    print("\n===== ORDER-EXACT TALLY vs ORT fp32 =====")
    for t in summary:
        n = sum(1 for r in summary[t] if r["exact_vs_ort"])
        bad = [r["idx"] for r in summary[t] if not r["exact_vs_ort"]]
        print("  %-14s %2d/16 exact_order   mismatched rows: %s" % (t, n, bad))

    # extra: pairwise vs fp16 (graph)
    if "fp16" in summary:
        print("\n===== qdq vs fp16(graph): rows where the two agree on live order =====")
        for t in summary:
            if t == "fp16":
                continue
            n = sum(1 for a, b in zip(summary[t], summary["fp16"]) if a["order"] == b["order"])
            print("  %-14s %2d/16 agree with fp16" % (t, n))

    with open(os.path.join(BASE, "qdq/out/compare_qdq_summary.json"), "w") as f:
        json.dump({"ref_orders": ref_ord, "variants": summary}, f, indent=1)
    print("\n### wrote qdq/out/compare_qdq_summary.json")

if __name__ == "__main__":
    main()
