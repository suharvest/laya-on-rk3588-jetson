#!/usr/bin/env python3
"""Compare round-1 (all-GEMM output Q/DQ) vs round-2 (residual GEMM output Q/DQ removed)
TensorRT engine layer info JSONs.

Usage: rs_layerinfo.py <round1.json> <round2.json>
"""
import json, sys
from collections import Counter, OrderedDict

def load(p):
    with open(p) as f:
        return json.load(f).get("Layers", [])

def dts(l):
    x = [i.get("Format/Datatype", "?") for i in l.get("Inputs", [])]
    x += [o.get("Format/Datatype", "?") for o in l.get("Outputs", [])]
    return x

def summarize(tag, layers):
    c = Counter()
    for l in layers:
        c[l.get("LayerType", "?")] += 1
    hist = Counter()
    for l in layers:
        for d in dts(l):
            hist[d] += 1
    n8 = sum(1 for l in layers if any("Int8" in str(x) for x in dts(l)))
    print("### %s: layers=%d  int8-layers=%d (%.1f%%)" % (tag, len(layers), n8, 100.0*n8/max(1,len(layers))))
    print("###   types=%s" % dict(c))
    print("###   dt=%s" % dict(hist))
    return c, hist

def gemm_tactics(tag, layers):
    g = [l for l in layers if l.get("LayerType") == "gemm"]
    i8 = [l for l in g if all("Int8" in str(d) for d in
                              [i.get("Format/Datatype") for i in l.get("Inputs", [])])]
    tac = Counter()
    for l in g:
        t = l.get("TacticName", "")
        k = "INT8" if "_i8i8_" in t else ("FP16" if "fp16" in t or "h1688" in t or "f16" in t else "other/empty")
        tac[k] += 1
    print("\n### %s GEMM layers=%d  all-int8-inputs=%d" % (tag, len(g), len(i8)))
    print("###   tactic classes=%s" % dict(tac))
    ts = Counter(l.get("TacticName", "")[:70] for l in g)
    for k, v in ts.most_common(12):
        print("      %3d  %s" % (v, k))
    return g

def named(tag, layers, key):
    hit = [l for l in layers if key in (l.get("Name") or "")]
    print("\n--- %s: layers whose Name contains %r : %d" % (tag, key, len(hit)))
    for l in hit[:25]:
        print("   %-58s type=%-8s ins=%s outs=%s" % (
            (l.get("Name") or "")[:58], l.get("LayerType"),
            [i.get("Format/Datatype") for i in l.get("Inputs", [])],
            [o.get("Format/Datatype") for o in l.get("Outputs", [])]))
    return hit

def main():
    a = load(sys.argv[1]); b = load(sys.argv[2])
    summarize("ROUND1 " + sys.argv[1], a)
    summarize("ROUND2 " + sys.argv[2], b)
    gemm_tactics("ROUND1", a)
    gemm_tactics("ROUND2", b)

    for key in ("ResTra", "Add", "Reshape_Transpose"):
        named("ROUND2", b, key)

    # residual-ish fusion layers in both
    print("\n--- layers in ROUND2 whose name starts with __myl_ (first 40) ---")
    myl = [l for l in b if (l.get("Name") or "").startswith("__myl_")]
    print("count:", len(myl))
    for l in myl[:40]:
        print("   %-70s type=%-8s ins=%s" % ((l.get("Name") or "")[:70], l.get("LayerType"),
              [i.get("Format/Datatype") for i in l.get("Inputs", [])]))

if __name__ == "__main__":
    main()
