#!/usr/bin/env python3
"""Per-row live-class ordering: round-2 residual-safe INT8 vs round-1 INT8 vs TRT fp16 vs ORT fp32.

Live entries = marker_mask == True. Masked positions carry the -10000 sentinel and would
dominate any norm, so rel_l2 is computed on live entries only (same rule as qdq_compare.py).
"""
import glob, json, os, sys
import numpy as np

BASE = "/home/harvest/laya-trt"
HELD = os.path.join(BASE, "calib_s512")
ORT = os.path.join(BASE, "int8/ortdump.s512_ng.json")

VARIANTS = [
    ("fp16",         "qdq/out/rank_fp16_nx_graph.json"),
    ("qdq1",         "qdq/out/rank_qdq_nx_graph.json"),
    ("rs_nx",        "qdq/out/rank_rs_nx_graph.json"),
    ("rs_nx_nogr",   "qdq/out/rank_rs_nx_nograph.json"),
    ("rs_nano",      "qdq/out/rank_rs_nano_graph.json"),
    ("rs_nano_nogr", "qdq/out/rank_rs_nano_nograph.json"),
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
    V, missing = {}, []
    for tag, p in VARIANTS:
        fp = os.path.join(BASE, p)
        if os.path.exists(fp):
            V[tag] = load(fp)
        else:
            missing.append((tag, fp))
    print("### ORT ref samples=%d  variants=%s  missing=%s" % (len(ref["samples"]), list(V), missing))

    ref_live = [live(s["logits"], m) for s, m in zip(ref["samples"], masks)]
    ref_ord = [order(x) for x in ref_live]

    summary = {}
    for tag, d in V.items():
        rows = []
        for i, (s, m) in enumerate(zip(d["samples"], masks)):
            lv = live(s["logits"], m)
            o = order(lv)
            rows.append({"idx": s["idx"], "live": int(m.sum()), "order": o,
                         "exact_vs_ort": bool(o == ref_ord[i]),
                         "rel_l2": rel_l2(lv, ref_live[i]) if len(lv) == len(ref_live[i]) else float("nan"),
                         "raw": [float(x) for x in np.asarray(s["logits"], dtype=np.float64).reshape(-1)]})
        summary[tag] = rows

    print("\n===== PER-ROW LIVE ORDER (16 heldout) =====")
    print("row  nl  ORT-ref" + "".join("%24s" % t for t in summary))
    for i in range(len(ref["samples"])):
        line = "%3d %3d  %-18s" % (i, int(masks[i].sum()), ref_ord[i])
        for t in summary:
            r = summary[t][i]
            line += "  %-21s %s" % (r["order"], "=" if r["exact_vs_ort"] else "X")
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

    for ref_tag in ("fp16", "qdq1"):
        if ref_tag in summary:
            print("\n===== %s vs %s: rows where the two agree on live order =====" % (ref_tag, "others"))
            for t in summary:
                if t == ref_tag:
                    continue
                n = sum(1 for a, b in zip(summary[t], summary[ref_tag]) if a["order"] == b["order"])
                print("  %-14s %2d/16 agree with %s" % (t, n, ref_tag))

    # raw logits for the live entries, all variants -- for the "still saturating?" question
    print("\n===== RAW LIVE LOGITS (per row, all variants) =====")
    for i in range(len(ref["samples"])):
        m = masks[i]
        print("row %2d  ORT   %s" % (i, np.round(ref_live[i], 4).tolist()))
        for t in summary:
            lv = live(summary[t][i]["raw"], m)
            print("        %-12s %s" % (t, np.round(lv, 4).tolist()))

    out = os.path.join(BASE, "qdq/out/compare_rs_summary.json")
    with open(out, "w") as f:
        json.dump({"ref_orders": ref_ord,
                   "variants": {t: [{k: v for k, v in r.items() if k != "raw"} for r in rows]
                                for t, rows in summary.items()}}, f, indent=1)
    print("\n### wrote %s" % out)

if __name__ == "__main__":
    main()
