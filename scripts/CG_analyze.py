"""CUDA-graph probe analysis: trtexec times --exportTimes, runner bench, 16-sample live ranking."""
import glob
import json
import os

T = "/Users/harvest/project/.cache/tasks"
GO = os.path.join(T, "graphout")
ORT = {
    512: os.path.join(T, "prod-results/ortdump.oldship_s512.nanfix.json"),
    90: os.path.join(T, "graphout/ortdump.newlin_s90.json"),
}
ORT_NOTE = {
    512: "oldship.s512.m16.nanfix.onnx (different lineage than the device engine = laya-multilingual.s512.m16.fp32.ng.onnx)",
    90: "laya-multilingual.s90.m16.newlin.fp32.onnx (same ONNX the s90 engine was built from)",
}
SENT = -10000.0


def pct(vals, q):
    s = sorted(vals)
    return s[min(len(s) - 1, int(q * len(s)))]


def live_order(logits):
    idx = [i for i, v in enumerate(logits) if v > SENT + 1.0]
    order = sorted(idx, key=lambda i: -logits[i])
    return idx, order


def kendall(a, b):
    n, conc, disc = len(a), 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            s = (a[i] - a[j]) * (b[i] - b[j])
            if s > 0:
                conc += 1
            elif s < 0:
                disc += 1
    return 1.0 if n < 2 else (conc - disc) / (n * (n - 1) / 2.0)


def ranks(a_order, b_order):
    """compare two orderings of the same index set (lists of marker indices)"""
    return a_order == b_order


def load(p):
    with open(p) as f:
        return json.load(f)


print("=" * 100)
print("STEP 1  trtexec  --exportTimes  (same engine, only --useCudaGraph differs)")
print("=" * 100)
import re


def logstats(dev, S, g, label):
    p = os.path.join(GO, dev, "graph", "trtexec_s%d_%s.log" % (S, g))
    if not os.path.exists(p):
        return None
    txt = open(p, errors="replace").read()
    for line in txt.splitlines():
        if "] [I] %s:" % label in line:
            return {k: float(v) for k, v in re.findall(
                r"(min|max|mean|median|percentile\(90%\)|percentile\(95%\)|percentile\(99%\))\s*=\s*([0-9.]+)",
                line)}
    return None


for dev in ("nx", "nano"):
    for S in (512, 90):
        for g in ("nograph", "graph"):
            p = os.path.join(GO, dev, "graph", "trtexec_s%d_%s.times.json" % (S, g))
            n = len(load(p)) if os.path.exists(p) else 0
            st = logstats(dev, S, g, "Latency")
            gs = logstats(dev, S, g, "GPU Compute Time")
            if st:
                print("  %-5s s%-4d %-7s n=%-4d  Latency      min=%.3f p50=%.3f p95=%.3f p99=%.3f max=%.3f mean=%.3f"
                      % (dev, S, g, n, st["min"], st["median"], st["percentile(95%)"],
                         st["percentile(99%)"], st["max"], st["mean"]))
                print("  %-5s s%-4d %-7s        GPUCompute   min=%.3f p50=%.3f p95=%.3f p99=%.3f max=%.3f mean=%.3f"
                      % (dev, S, g, gs["min"], gs["median"], gs["percentile(95%)"],
                         gs["percentile(99%)"], gs["max"], gs["mean"]))
                if g == "graph":
                    sn = logstats(dev, S, "nograph", "Latency")
                    gn = logstats(dev, S, "nograph", "GPU Compute Time")
                    print("        -> Latency    p50 speedup %.3fx   p95 speedup %.3fx"
                          % (sn["median"] / st["median"], sn["percentile(95%)"] / st["percentile(95%)"]))
                    print("        -> GPUCompute p50 speedup %.3fx   p95 speedup %.3fx"
                          % (gn["median"] / gs["median"], gn["percentile(95%)"] / gs["percentile(95%)"]))
            else:
                print("  %-5s s%-4d %-7s n=%-4d MISSING log" % (dev, S, g, n))
            if os.path.exists(p):
                lat = [r["latencyMs"] for r in load(p)]
                cmp_ = [r["computeMs"] for r in load(p)]
                print("  %-5s s%-4d %-7s        fromJson[latencyMs] n=%d p50=%.3f p95=%.3f min=%.3f mean=%.3f | "
                      "computeMs p50=%.3f" % (dev, S, g, len(lat), pct(lat, .5), pct(lat, .95),
                                              min(lat), sum(lat) / len(lat), pct(cmp_, .5)))
        print()

print("=" * 100)
print("STEP 2  capture / instantiate")
print("=" * 100)
for dev in ("nx", "nano"):
    for f in sorted(glob.glob(os.path.join(GO, dev, "graph", "*.json"))):
        b = os.path.basename(f)
        if b.startswith("trtexec") or b.startswith("results"):
            continue
        d = load(f)
        cap = {k: d.get(k) for k in ("graph", "capture_mode", "graph_nodes", "graph_root_nodes",
                                     "instantiate_ms", "instantiate_form", "instantiate_ret_arity",
                                     "precondition", "error")}
        print("  %-5s %-22s %s" % (dev, b, json.dumps(cap)))
print()

print("=" * 100)
print("STEP 3  runner latency  (same engine / inputs / protocol; --iters as stated)")
print("=" * 100)
for dev in ("nx", "nano"):
    for S in (512, 90):
        for g in ("nograph", "graph"):
            p = os.path.join(GO, dev, "graph", "bench_s%d_%s.json" % (S, g))
            if not os.path.exists(p):
                print("  %-5s s%-4d %-7s MISSING" % (dev, S, g))
                continue
            d = load(p)
            print("  %-5s s%-4d %-7s ok=%-5s %s" % (dev, S, g, d.get("ok"), json.dumps(d.get("latency_ms"))))
        a = os.path.join(GO, dev, "graph", "bench_s%d_nograph.json" % S)
        b = os.path.join(GO, dev, "graph", "bench_s%d_graph.json" % S)
        if os.path.exists(a) and os.path.exists(b):
            la, lb = load(a), load(b)
            if la.get("latency_ms") and lb.get("latency_ms"):
                ra = la["latency_ms"]["p50"] / lb["latency_ms"]["p50"]
                r9 = la["latency_ms"]["p95"] / lb["latency_ms"]["p95"]
                print("        -> p50 speedup %.3fx   p95 speedup %.3fx" % (ra, r9))
        print()

print("=" * 100)
print("STEP 4  16-sample live ranking:  graph vs non-graph vs ORT fp32")
print("=" * 100)
for dev in ("nx", "nano"):
    for S in (512, 90):
        pn = os.path.join(GO, dev, "graph", "rank_s%d_nograph.json" % S)
        pg = os.path.join(GO, dev, "graph", "rank_s%d_graph.json" % S)
        if not (os.path.exists(pn) and os.path.exists(pg)):
            print("  %-5s s%-4d MISSING rank json" % (dev, S))
            continue
        dn, dg = load(pn), load(pg)
        if not (dn.get("ok") and dg.get("ok")):
            print("  %-5s s%-4d not ok: nograph=%s graph=%s" % (dev, S, dn.get("error"), dg.get("error")))
            continue
        o = load(ORT[S])
        om = {int(s["idx"]): s for s in o["samples"]}
        print("  --- %s s%d   ORT ref: %s" % (dev, S, ORT_NOTE[S]))
        print("  %-4s %-3s %-24s %-24s %-24s %-8s %-8s %-8s %s"
              % ("idx", "nl", "ORT order", "non-graph order", "graph order",
                 "g==ng", "g==ort", "ng==ort", "identical_bits_g_vs_ng"))
        n_ge_ng = n_g_ort = n_ng_ort = n_bits = 0
        kt_ort = []
        first = None
        for k in range(len(dn["samples"])):
            ln = dn["samples"][k]["logits"]
            lg = dg["samples"][k]["logits"]
            lo = om[k]["logits"]
            _, on = live_order(ln)
            _, og = live_order(lg)
            _, oo = live_order(lo)
            bits = ln == lg if ln == lg else [a == b for a, b in zip(ln, lg)]
            same_bits = all(abs(a - b) == 0 for a, b in zip(ln, lg))
            if og == on:
                n_ge_ng += 1
            if og == oo:
                n_g_ort += 1
            if on == oo:
                n_ng_ort += 1
            if same_bits:
                n_bits += 1
            kt_ort.append(kendall([lo[i] for i in og], [lo[i] for i in oo]) if len(og) == len(oo) else 0.0)
            if first is None:
                first = (k, len(on))
            print("  %-4d %-3d %-24s %-24s %-24s %-8s %-8s %-8s %s"
                  % (k, len(on), oo, on, og,
                     "yes" if og == on else "NO", "yes" if og == oo else "NO",
                     "yes" if on == oo else "NO", "yes" if same_bits else "NO"))
        print("        totals: graph==nongraph order %d/%d   graph==ORT %d/%d   nongraph==ORT %d/%d   "
              "bitwise_identical %d/%d   kendall(g,ORT) mean %.4f min %.4f"
              % (n_ge_ng, len(dn["samples"]), n_g_ort, len(dn["samples"]),
                 n_ng_ort, len(dn["samples"]), n_bits, len(dn["samples"]),
                 sum(kt_ort) / len(kt_ort), min(kt_ort)))
        # top1 (argmax over live markers)
        t1_g = t1_ng = t1_ort = 0
        for k in range(len(dn["samples"])):
            _, og = live_order(dg["samples"][k]["logits"])
            _, on = live_order(dn["samples"][k]["logits"])
            _, oo = live_order(om[k]["logits"])
            t1_g += og[0] == oo[0]
            t1_ng += on[0] == oo[0]
            t1_ort += 1
        print("        top1 vs ORT: graph %d/%d  non-graph %d/%d" % (t1_g, t1_ort, t1_ng, t1_ort))
        print()

print("=" * 100)
print("df / versions")
print("=" * 100)
for dev in ("nx", "nano"):
    p = os.path.join(GO, dev, "graph", "df_%s.txt" % dev)
    print("  %-5s %s" % (dev, open(p).read().strip().replace("\n", " | ") if os.path.exists(p) else "MISSING"))
