"""Does this graph carry the HF NaN guard, and does any guard sit between a Softmax and a MatMul?

The rknn-toolkit2 SDPA fusion matches `MatMul -> [Mul|Add|Div]* -> Softmax -> MatMul` and needs
the Softmax to feed the final MatMul directly. The guard is `Where(cond, x, y)` inserted between
them, which breaks the match. This prints, per graph:

  * total node count and the op histogram for IsNaN / Not / Equal / Where
  * every `Where` whose output feeds exactly one node, together with that consumer's op type
  * a flag for whether any of the Where inputs traces back to a Softmax within a few hops

That is the same predicate 81_guard_report.py uses, kept here as a self-contained copy so this
task does not depend on that script's CLI.
"""
import argparse
import collections
import sys

import onnx

SOFTMAX_OPS = {"Softmax", "exSoftmax13", "Softmax13"}
MATMUL_OPS = {"MatMul", "Gemm"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--max-print", type=int, default=5)
    args = ap.parse_args()

    g = onnx.load(args.onnx, load_external_data=False).graph
    prod = {}
    for n in g.node:
        for o in n.output:
            prod[o] = n
    cons = collections.defaultdict(list)
    for n in g.node:
        for i in n.input:
            cons[i].append(n)

    ops = collections.Counter(n.op_type for n in g.node)
    print("### FILE %s nodes=%d" % (args.onnx, len(g.node)), flush=True)
    print("### op histogram (attention-relevant): %s"
          % {k: ops[k] for k in ("MatMul", "Softmax", "IsNaN", "Not", "Equal", "Where", "Transpose")
             if ops.get(k)}, flush=True)

    wheres = [n for n in g.node if n.op_type == "Where"]
    print("### Where nodes total: %d" % len(wheres), flush=True)

    def hops_to_softmax(t, depth=6):
        cur = t
        chain = []
        for _ in range(depth):
            p = prod.get(cur)
            if p is None:
                chain.append("INPUT/CONST(%s)" % cur)
                return False, chain
            chain.append("%s" % p.op_type)
            if p.op_type in SOFTMAX_OPS:
                return True, chain
            if not p.input:
                return False, chain
            cur = p.input[1] if (p.op_type == "Where" and len(p.input) > 1) else p.input[0]
        return False, chain

    between = []
    for n in wheres:
        out_cons = cons[n.output[0]]
        softmax_src = False
        for t in n.input[1:]:
            hit, _ = hops_to_softmax(t)
            softmax_src = softmax_src or hit
        rec = {"out": n.output[0], "n_consumers": len(out_cons),
               "consumer_ops": [c.op_type for c in out_cons],
               "single_mul_consumer": (len(out_cons) == 1 and out_cons[0].op_type in MATMUL_OPS),
               "softmax_origin": softmax_src}
        if rec["softmax_origin"] and len(out_cons) == 1 and out_cons[0].op_type in MATMUL_OPS:
            between.append(rec)
    print("### Where feeding exactly one MatMul: %d" % sum(
        1 for n in wheres if len(cons[n.output[0]]) == 1 and cons[n.output[0]][0].op_type in MATMUL_OPS),
        flush=True)
    print("### GUARDS BLOCKING SDPA FUSION (softmax-origin Where -> single MatMul): %d"
          % len(between), flush=True)
    for r in between[:args.max_print]:
        print("###   %s" % r, flush=True)
    print("### Verdict: %s" % ("GUARDED (fusion blocked without surgery)" if between
                               else "GUARD-FREE (fusion can match as-is)"), flush=True)
    print("### A2_GUARD_CHECK_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
