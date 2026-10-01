#!/usr/bin/env python3
"""Full analysis: per-node profile, medians, graph vs non-graph, s90->s512 scaling, nano cross-check."""
import json, re, os, statistics as st

D = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(D, "prof")
G = os.path.join(D, "graph")
PN = os.path.join(D, "prof_nano")
GN = os.path.join(D, "graph_nano")

def load(tag, base=P, pre="prof_"):
    d = json.load(open(os.path.join(base, "%s%s.json" % (pre, tag))))
    hdr, rows = d[0], [(e["name"], e["averageMs"], e["medianMs"]) for e in d[1:]]
    return hdr, rows

def total_from_log(p):
    for line in reversed(open(p).read().splitlines()):
        m = re.search(r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+Total\s*$", line)
        if m:
            return float(m.group(2)), float(m.group(3))
    return None

def unprofiled(f):
    d = json.load(open(f))
    c = [x["computeMs"] for x in d]
    return len(d), st.mean(c), st.median(c), min(c), max(c)

def canon(n):
    n = re.sub(r"_myl0_\d+$", "", n)
    n = re.sub(r"_mye\d+_myl0_\d+$", "", n)
    n = re.sub(r"_mye\d+$", "", n)
    return n

def layer_prefix(n):
    m = re.search(r"/encoder/layers_(\d+)/", n)
    if m:
        return "encoder.layers_%02d" % int(m.group(1))
    m = re.search(r"(?<!/encoder)/layers_(\d+)/", n)
    if m:
        return "decoder.layers_%02d" % int(m.group(1))
    return "head/emb/pre-post"

CATS = [
    ("attn_core_SDPA",    lambda n: n.startswith("_gemm_mha_v2")),
    ("attn_qkv_gemm",     lambda n: "/attn/Wqkv/MatMul" in n),
    ("attn_out_proj",     lambda n: "/attn/Wo/MatMul" in n),
    ("attn_qkv_dec",      lambda n: "self_attn/MatMul" in n),
    ("rotary_matmul",     lambda n: "rotary_emb" in n),
    ("rotary_apply",      lambda n: "TraConSinCos" in n or "TraSliSli" in n),
    ("ffn_gemm",          lambda n: "/mlp/Wo/MatMul" in n or "/linear1/MatMul" in n or "/linear2/MatMul" in n),
    ("ffn_fused_fc",      lambda n: bool(re.match(r"__myl_Fc", n)) or bool(re.match(r"__mye\d+", n))),
    ("norm_residual_ptw", lambda n: "AddCasMea" in n or bool(re.match(r"__myl_CasMea", n))),
    ("embed_ptw",         lambda n: n.startswith("__myl_CasSum") or n.startswith("__myl_CasGatCasMea")),
    ("mask_ptw",          lambda n: "CasRepOrResNot" in n or "CasNotSelResSelResEql" in n),
    ("head_ptw",          lambda n: any(n.startswith("__myl_" + p) for p in
                           ("MaxMinResCasRepGat", "SliRes", "MulSum", "NotAddResSelMaxSubExpSum",
                            "DivMulTop", "MaxMinLogMulSum", "MaxMinCasLogDiv", "ResMulSumAdd",
                            "DivCasErfCasAddMulMulResMulSumAdd", "CasGatResAdd", "Add"))),
    ("reformat_copy",     lambda n: n.startswith("Reformatting CopyNode")),
    ("aux_shape",         lambda n: n.startswith("dummy_shape_call")),
    ("other",             lambda n: True),
]
def cat(n):
    for c, f in CATS:
        if f(n):
            return c
    return "other"

TAGS = ["s512_graph", "s512_nograph", "s90_graph", "s90_nograph"]
NTAGS = ["nano_s512_graph", "nano_s512_nograph"]

print("=" * 108)
print("S0 — RAW PROFILE SUMMARY + node-sum vs INDEPENDENT end-to-end (same protocol / other tool)")
print("=" * 108)
print("%-17s %5s %10s %10s %12s %12s %9s %9s" % (
    "tag", "nodes", "sum_avg", "sum_med", "unprof_mean", "unprof_p50", "sum/mean", "sum/p50"))
def alltags():
    out = []
    for t in TAGS:
        out.append((t, load(t), total_from_log(os.path.join(P, "prof_%s.log" % t)),
                    unprofiled(os.path.join(G, "trtexec_%s.times.json" % t))))
    for t in NTAGS:
        b = t.replace("nano_", "")
        out.append(("nano." + b, load(b, PN, "prof_"), total_from_log(os.path.join(PN, "prof_%s.log" % b)),
                    unprofiled(os.path.join(GN, "trtexec_%s.times.json" % b))))
    return out
ALL = alltags()
for t, (hdr, rows), tot, (n, m, p50, mn, mx) in ALL:
    sa = sum(r[1] for r in rows); sm = sum(r[2] for r in rows)
    print("%-17s %5d %10.4f %10.4f %12.4f %12.4f %9.3f %9.3f   %s" % (
        t, len(rows), sa, sm, m, p50, sa / m, sa / p50, "NANO" if "nano" in t or t in ("s512_graph","s512_nograph") and False else ""))

print()
print("=" * 108)
print("S1 — operator-category aggregation (share of profile Total)")
print("=" * 108)
for t, (hdr, rows), tot, up in ALL:
    ta, tm = tot
    byc = {}
    for n_, a, m_ in rows:
        c = cat(n_); v = byc.setdefault(c, [0.0, 0.0, 0])
        v[0] += a; v[1] += m_; v[2] += 1
    print("\n--- %s  (profile Total mean=%.4f median=%.4f ms) ---" % (t, ta, tm))
    print("%-22s %10s %8s %10s %8s %5s" % ("category", "sum_avg", "%avg", "sum_med", "%med", "n"))
    for c, (a, m_, k) in sorted(byc.items(), key=lambda kv: -kv[1][1]):
        print("%-22s %10.4f %7.2f%% %10.4f %7.2f%% %5d" % (c, a, 100 * a / ta, m_, 100 * m_ / tm, k))

print()
print("=" * 108)
print("S2 — layer-prefix aggregation (encoder.layers_00..21 / decoder.layers_0..1 / head-emb)")
print("=" * 108)
for t, (hdr, rows), tot, up in ALL:
    ta, tm = tot
    byp = {}
    for n_, a, m_ in rows:
        p = layer_prefix(n_); v = byp.setdefault(p, [0.0, 0.0, 0])
        v[0] += a; v[1] += m_; v[2] += 1
    enc = sorted(k for k in byp if k.startswith("encoder."))
    print("\n--- %s ---" % t)
    if enc:
        vals = [byp[k][0] for k in enc]
        print("  encoder.layers: %d groups  sum_avg=%.4f ms (%.2f%% of Total %.4f)  min=%.4f max=%.4f mean=%.4f" % (
            len(enc), sum(vals), 100 * sum(vals) / ta, ta, min(vals), max(vals), st.mean(vals)))
    for k in sorted(byp):
        if not k.startswith("encoder."):
            a, m_, c = byp[k]
            print("  %-18s sum_avg=%8.4f (%6.2f%%)  sum_med=%8.4f  n=%d" % (k, a, 100 * a / ta, m_, c))
    print("  per-encoder-layer detail (sum_avg ms): %s" % ", ".join(
        "L%s=%.3f" % (k.split("_")[-1], byp[k][0]) for k in enc))

print()
print("=" * 108)
print("S3 — Top-20 nodes by averageMs (and by medianMs)")
print("=" * 108)
for t, (hdr, rows), tot, up in ALL:
    ta, tm = tot
    print("\n--- %s (Total mean %.4f) ---" % (t, ta))
    for n_, a, m_ in sorted(rows, key=lambda r: -r[1])[:20]:
        print("  avg %7.4f  med %7.4f  %5.2f%%  %s" % (a, m_, 100 * a / ta, n_[:96]))

print()
print("=" * 108)
print("S4 — graph vs non-graph, per-node delta, MEDIAN based (outlier robust) and AVERAGE based")
print("=" * 108)
for seq in ["512", "90"]:
    tg = "s%s_graph" % seq; tn = "s%s_nograph" % seq
    _, rg = load(tg); _, rn = load(tn)
    ag, an = total_from_log(os.path.join(P, "prof_%s.log" % tg))[0], total_from_log(os.path.join(P, "prof_%s.log" % tn))[0]
    dg = {r[0]: (r[1], r[2]) for r in rg}; dn = {r[0]: (r[1], r[2]) for r in rn}
    common = [k for k in dg if k in dn]
    print("\n### s%s  named-nodes=%d  Total(mean): graph=%.4f non-graph=%.4f  delta=%.4f ms (%.2f%%)" % (
        seq, len(common), ag, an, an - ag, 100 * (an - ag) / an))
    for idx, lab in ((1, "MEDIAN"), (0, "AVERAGE")):
        deltas = sorted(((k, dn[k][idx] - dg[k][idx], dg[k][idx], dn[k][idx]) for k in common), key=lambda x: -x[1])
        pos = sum(1 for k in common if deltas and dn[k][idx] - dg[k][idx] > 0.0005)
        neg = sum(1 for k in common if dg[k][idx] - dn[k][idx] > 0.0005)
        tot = sum(d[1] for d in deltas)
        print("  [%s] net=%.4f ms | nodes faster=%.1f%%  slower=%.1f%%  unchanged(<=0.5us)=%.1f%%  | sum of all deltas=%.4f" % (
            lab, tot, 100.0 * pos / len(common), 100.0 * neg / len(common),
            100.0 * (len(common) - pos - neg) / len(common), tot))
        print("    top-10 FASTER under graph:  " + " | ".join("%s -%.4f" % (k[:44], d) for k, d, g, n_ in deltas[:10]))
        print("    top-5  SLOWER under graph:  " + " | ".join("%s +%.4f" % (k[:44], -d) for k, d, g, n_ in deltas[-5:]))
        # aggregate the <5us nodes vs >=5us nodes
        small = [d for k, d, g, n_ in deltas if g < 0.005]; big = [d for k, d, g, n_ in deltas if g >= 0.005]
        print("    nodes with graph-time <5us: n=%d sum_delta=%.4f | >=5us: n=%d sum_delta=%.4f" % (
            len(small), sum(small), len(big), sum(big)))

print()
print("=" * 108)
print("S5 — s90 -> s512 per-canonical-node scaling (seq x5.689), MEDIAN based")
print("=" * 108)
_, r1 = load("s90_graph"); _, r2 = load("s512_graph")
m1, m2 = {}, {}
for n_, a, m_ in r1:
    m1.setdefault(canon(n_), []).append(m_)
for n_, a, m_ in r2:
    m2.setdefault(canon(n_), []).append(m_)
rows = []
for k in m1:
    if k in m2:
        a, b = st.median(m1[k]), st.median(m2[k])
        if a > 0.002 and b > 0.002:
            rows.append((k, a, b, b / a, len(m1[k]), len(m2[k])))
print("matched canonical nodes (both median >0.002ms): %d" % len(rows))
cats = {}
for k, a, b, s, c1, c2 in rows:
    cats.setdefault(cat(k), []).append(s)
print("\nby CATEGORY: median / min / max scaling factor")
for c, v in sorted(cats.items(), key=lambda kv: -st.median(kv[1])):
    print("  %-22s n=%3d  median=%5.2fx  min=%5.2fx  max=%5.2fx" % (c, len(v), st.median(v), min(v), max(v)))
print("\ntop-20 superlinear nodes:")
for k, a, b, s, c1, c2 in sorted(rows, key=lambda r: -r[3])[:20]:
    print("  %6.2fx   s90 %8.4f -> s512 %8.4f   n=%d->%d  %s" % (s, a, b, c1, c2, k[:70]))
print("\ntop-10 sublinear nodes:")
for k, a, b, s, c1, c2 in sorted(rows, key=lambda r: r[3])[:10]:
    print("  %6.2fx   s90 %8.4f -> s512 %8.4f   n=%d->%d  %s" % (s, a, b, c1, c2, k[:70]))

print()
print("=" * 108)
print("S6 — independent end-to-end p50 (laya_trt_run_graph.py bench) on both devices")
print("=" * 108)
for base, dev in ((G, "orin-nx"), (GN, "orin-nano")):
    for f in sorted(os.listdir(base)):
        if f.startswith("bench_") and f.endswith(".json"):
            d = json.load(open(os.path.join(base, f)))
            L = d.get("latency_ms") or {}
            print("%-10s %-26s graph=%-5s n=%-4s p50=%-8.3f p95=%-8.3f min=%-8.3f mean=%.3f" % (
                dev, f, d.get("graph"), L.get("n"), L.get("p50", -1), L.get("p95", -1), L.get("min", -1), L.get("mean", -1)))

print()
print("=" * 108)
print("S7 — kernel-count / per-kernel budget for s512_graph")
print("=" * 108)
_, rows = load("s512_graph")
ta, tm = total_from_log(os.path.join(P, "prof_s512_graph.log"))
vals = sorted((r[2] for r in rows), reverse=True)
print("nodes=%d  sum_med=%.4f  mean_per_node=%.4f ms  median_per_node=%.4f ms" % (
    len(vals), sum(vals), sum(vals) / len(vals), st.median(vals)))
for k in (1, 5, 10, 25, 50, 100):
    print("  top-%3d nodes account for %.4f ms = %.1f%% of Total" % (k, sum(vals[:k]), 100 * sum(vals[:k]) / ta))
print("  nodes with median < 5us: n=%d  (sum=%.4f ms, %.1f%%)" % (
    sum(1 for v in vals if v < 0.005), sum(v for v in vals if v < 0.005), 100 * sum(v for v in vals if v < 0.005) / ta))
print("  nodes with median <10us: n=%d  (sum=%.4f ms, %.1f%%)" % (
    sum(1 for v in vals if v < 0.010), sum(v for v in vals if v < 0.010), 100 * sum(v for v in vals if v < 0.010) / ta))
