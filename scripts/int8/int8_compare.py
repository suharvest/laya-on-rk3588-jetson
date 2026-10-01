"""Per-row live ranking comparison: INT8 PTQ vs TRT fp16 vs ORT fp32.

The acceptance rule is per-row live ordering, not a norm: the model's answer is the order of the
live (marker_mask == True) logits, so that is what is compared row by row. rel_l2 is reported as
a secondary magnitude check, live entries only -- the -10000 masked sentinels would otherwise
dominate any norm and hide a collapsed live head.
"""
import glob
import json
import os
import sys

import numpy as np

BASE = "/home/harvest/laya-trt"
HELD = os.path.join(BASE, "calib_s512")
ORT = os.path.join(BASE, "int8/ortdump.s512_ng.json")


def load(p):
    with open(p) as f:
        return json.load(f)


def live(logits, mask):
    return np.asarray(logits, dtype=np.float64).reshape(-1)[mask]


def order(x):
    return list(int(v) for v in np.argsort(-x, kind="stable"))


def rel_l2(a, b):
    d = np.linalg.norm(a - b)
    n = np.linalg.norm(b)
    return float(d / n) if n > 0 else float("nan")


def main():
    dirs = sorted(glob.glob(os.path.join(HELD, "sample_*")))
    masks = [np.load(os.path.join(d, "marker_mask.npy")).reshape(-1).astype(bool) for d in dirs]
    ref = load(ORT)
    summary = {}

    variants = {}
    for tag, p in (("fp16_graph", "graph/rank_s512_graph.json"),
                   ("fp16_nograph", "graph/rank_s512_nograph.json"),
                   ("int8_nx_graph", "int8/out/rank_int8_nx_graph.json"),
                   ("int8_nx_nograph", "int8/out/rank_int8_nx_nograph.json"),
                   ("int8_nano_graph", "int8/out/rank_int8_nano_graph.json"),
                   ("int8_nano_nograph", "int8/out/rank_int8_nano_nograph.json")):
        fp = os.path.join(BASE, p)
        if os.path.exists(fp):
            variants[tag] = load(fp)
        else:
            print("### missing %s" % fp)

    ref_orders = [order(live(s["logits"], m)) for s, m in zip(ref["samples"], masks)]
    ref_live = [live(s["logits"], m) for s, m in zip(ref["samples"], masks)]

    for tag, d in variants.items():
        if not d.get("ok"):
            print("### %s not ok: %s" % (tag, d.get("error")))
            summary[tag] = {"ok": False, "error": d.get("error")}
            continue
        o_exact, rows, rl = 0, [], []
        for k, (s, m) in enumerate(zip(d["samples"], masks)):
            x = live(s["logits"], m)
            o = order(x)
            eq = (o == ref_orders[k])
            o_exact += int(eq)
            rl.append(rel_l2(x, ref_live[k]))
            rows.append({"idx": k, "n_live": int(m.sum()),
                         "ort_order": ref_orders[k],
                         "var_order": o, "exact_vs_ort": bool(eq),
                         "ort_live": [round(float(v), 4) for v in ref_live[k]],
                         "var_live": [round(float(v), 4) for v in x],
                         "live_rel_l2_vs_ort": round(rl[-1], 6)})
        hist = {}
        for o in ref_orders:
            hist[tuple(o)] = hist.get(tuple(o), 0) + 1
        degraded = [r for r in rows if not np.isfinite(r["live_rel_l2_vs_ort"]) or r["live_rel_l2_vs_ort"] > 0.5]
        summary[tag] = {"ok": True, "order_exact_vs_ort": o_exact, "n": len(rows),
                        "live_rel_l2_mean": round(float(np.mean(rl)), 6),
                        "live_rel_l2_max": round(float(np.max(rl)), 6),
                        "latency_ms": d.get("latency_ms"),
                        "engine_size_mb": d.get("engine_size_mb"),
                        "rows_over_0.5_rel_l2": [r["idx"] for r in degraded],
                        "rows": rows}
        print("### %-16s order_exact_vs_ort %2d/%d  live_rel_l2 mean=%.5f max=%.5f  p50=%s ms"
              % (tag, o_exact, len(rows), np.mean(rl), np.max(rl),
                 d.get("latency_ms", {}).get("p50")))
        for r in rows:
            flag = "OK " if r["exact_vs_ort"] else "MISMATCH"
            print("    %s sim[%02d] n_live=%2d rel_l2=%.5f  ort=%s  %s=%s"
                  % (flag, r["idx"], r["n_live"], r["live_rel_l2_vs_ort"],
                     r["ort_order"], tag, r["var_order"]))
            if not r["exact_vs_ort"]:
                print("       ort_live=%s" % r["ort_live"])
                print("       %s_live=%s" % (tag, r["var_live"]))

    # fp16 and int8 must also be compared to each other row by row
    pairs = [("int8_nx_graph", "fp16_graph"), ("int8_nx_nograph", "fp16_nograph"),
             ("fp16_graph", "fp16_nograph"), ("int8_nx_graph", "int8_nx_nograph")]
    print()
    for a, b in pairs:
        if a not in variants or b not in variants:
            continue
        if not (variants[a].get("ok") and variants[b].get("ok")):
            continue
        same_order = 0
        bitwise = 0
        maxdiff = 0.0
        for ka, kb, m in zip(variants[a]["samples"], variants[b]["samples"], masks):
            xa = np.asarray(ka["logits"], dtype=np.float64).reshape(-1)
            xb = np.asarray(kb["logits"], dtype=np.float64).reshape(-1)
            if order(xa[m]) == order(xb[m]):
                same_order += 1
            if np.array_equal(np.asarray(ka["logits"]), np.asarray(kb["logits"])):
                bitwise += 1
            maxdiff = max(maxdiff, float(np.max(np.abs(xa - xb))))
        print("### %-16s vs %-16s order_equal %2d/16  logits_bitwise_equal %2d/16  max_absdiff=%.6g"
              % (a, b, same_order, bitwise, maxdiff))
        summary["%s_vs_%s" % (a, b)] = {"order_equal": same_order, "bitwise_equal": bitwise,
                                        "max_absdiff": maxdiff}

    with open(os.path.join(BASE, "int8/out/compare_int8.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("### wrote int8/out/compare_int8.json")


if __name__ == "__main__":
    main()
