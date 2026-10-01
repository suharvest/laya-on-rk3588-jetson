"""Per-sample ranking acceptance across any number of dumped model outputs.

Inputs are dump JSONs with {"samples": [{"idx", "logits", "act_logits"}]}, produced either by the
host simulator (scripts/43_convert.py --rank-dump) or by an on-device run (scripts/47_device_dump.py).
The reference is the ORT fp32 dump on the same rows (golden_s90/batch16_ort_nanfix_ref.json).

rel_l2 alone is not acceptance: a quantized model can track the fp32 logits in L2 and still change
which option wins, and this project has an int8 case (SenseVoice) where a good-looking rel_l2
shipped a model whose output was empty. So the printed primary number is the order of the live
(masked-in) markers, sample by sample.
"""
import argparse
import collections
import glob
import json
import os
import sys

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


def live_masks(heldout):
    out = []
    for d in sorted(glob.glob(os.path.join(heldout, "sample_*"))):
        m = os.path.join(d, "marker_mask.npy")
        out.append(np.load(m).reshape(-1).astype(bool) if os.path.exists(m) else None)
    return out


def load_dump(path):
    with open(path) as f:
        j = json.load(f)
    return {int(s["idx"]): s for s in j["samples"]}, j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", help="name=json for the fp32 reference")
    ap.add_argument("--dump", action="append", default=[], help="name=json, repeatable")
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    masks = live_masks(args.heldout)
    print("### heldout=%s samples=%d live_sizes=%s"
          % (args.heldout, len(masks),
             collections.Counter(int(m.sum()) if m is not None else -1 for m in masks)), flush=True)

    models = collections.OrderedDict()
    if args.ref:
        n, p = args.ref.split("=", 1)
        d, j = load_dump(p)
        models[n] = d
        print("### ref [%s] %s samples=%d" % (n, p, len(d)), flush=True)
    for kv in args.dump:
        n, p = kv.split("=", 1)
        if not os.path.exists(p):
            print("### MISSING dump [%s] %s" % (n, p), flush=True)
            continue
        d, j = load_dump(p)
        models[n] = d
        print("### model [%s] %s samples=%d meta=%s"
              % (n, p, len(d), {k: j[k] for k in ("quantized_dtype", "do_quantization", "target", "rknn")
                                if k in j}), flush=True)

    names = list(models)
    result = {"heldout": args.heldout, "models": {}, "pairs": {}}
    for n in names:
        result["models"][n] = models[n]

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            rows = []
            for idx in sorted(models[a]):
                s = models[a][idx]
                if idx not in models[b]:
                    continue
                t = models[b][idx]
                live = masks[idx] if idx < len(masks) and masks[idx] is not None else \
                    np.ones(len(s["logits"]), dtype=bool)
                x = np.asarray(s["logits"], dtype=np.float64).reshape(-1)
                y = np.asarray(t["logits"], dtype=np.float64).reshape(-1)
                xl, yl = x[live], y[live]
                ro = np.argsort(-xl, kind="stable")
                go = np.argsort(-yl, kind="stable")
                rows.append({
                    "idx": idx, "n_live": int(len(xl)),
                    "top1_match": bool(int(ro[0]) == int(go[0])),
                    "order_exact": bool(np.array_equal(ro, go)),
                    "kendall_tau": round(float(kendall(xl, yl)), 6),
                    "rel_l2": round(float(np.linalg.norm(xl - yl) / max(1e-12, np.linalg.norm(xl))), 6),
                    "max_abs_diff": round(float(np.max(np.abs(xl - yl))), 6),
                    "order_a": [int(v) for v in ro], "order_b": [int(v) for v in go],
                    "vals_a": [round(float(v), 5) for v in xl], "vals_b": [round(float(v), 5) for v in yl]})
            key = "%s_vs_%s" % (a, b)
            # act_logits is the secondary head output: reported, never an acceptance gate
            act = []
            for idx in sorted(models[a]):
                if idx not in models[b]:
                    continue
                xa = np.asarray(models[a][idx]["act_logits"], dtype=np.float64).reshape(-1)
                ya = np.asarray(models[b][idx]["act_logits"], dtype=np.float64).reshape(-1)
                act.append(float(np.linalg.norm(xa - ya) / max(1e-12, np.linalg.norm(xa))))
            n = len(rows)
            summ = {
                "n": n,
                "top1_match": "%d/%d" % (sum(r["top1_match"] for r in rows), n),
                "order_exact": "%d/%d" % (sum(r["order_exact"] for r in rows), n),
                "kendall_tau_mean": round(float(np.mean([r["kendall_tau"] for r in rows])), 6),
                "kendall_tau_min": round(float(np.min([r["kendall_tau"] for r in rows])), 6),
                "rel_l2_mean": round(float(np.mean([r["rel_l2"] for r in rows])), 6),
                "rel_l2_max": round(float(np.max([r["rel_l2"] for r in rows])), 6),
                "max_abs_diff_max": round(float(np.max([r["max_abs_diff"] for r in rows])), 6),
                "act_rel_l2_mean": round(float(np.mean(act)), 6) if act else None,
                "act_rel_l2_max": round(float(np.max(act)), 6) if act else None,
                "verdict": "PASS" if all(r["order_exact"] for r in rows) else
                           ("PARTIAL" if all(r["top1_match"] for r in rows) else "FAIL"),
            }
            result["pairs"][key] = {"summary": summ, "rows": rows}
            print("### %-34s top1=%s exact=%s tau_min=%.4f rel_l2_mean=%.6f rel_l2_max=%.6f "
                  "maxabs_max=%.5f act_rel_l2_max=%s -> %s"
                  % (key, summ["top1_match"], summ["order_exact"], summ["kendall_tau_min"],
                     summ["rel_l2_mean"], summ["rel_l2_max"], summ["max_abs_diff_max"],
                     summ["act_rel_l2_max"], summ["verdict"]), flush=True)
            for r in rows:
                print("    sample %02d n=%2d %s tau=%.4f rel_l2=%.6f\n      a order=%s vals=%s\n"
                      "      b order=%s vals=%s"
                      % (r["idx"], r["n_live"], "OK  " if r["order_exact"] else "DIFF",
                         r["kendall_tau"], r["rel_l2"], r["order_a"], r["vals_a"],
                         r["order_b"], r["vals_b"]), flush=True)

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print("### wrote", args.out, flush=True)
    print("### RANK_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
