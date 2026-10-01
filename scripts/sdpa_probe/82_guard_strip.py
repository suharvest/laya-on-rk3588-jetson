"""Delete the NaN-guard `Where` from every attention block so Softmax feeds MatMul directly.

The guard is `Where(Not(Equal(p,p)), 0, p)` sitting between the softmax output `p` and the
attention MatMul. The RKNN sdpa fusion rule requires the last two ops of the matched pattern
to be `softmax, matmul` with nothing in between, so this Where is the one op that breaks it.

Safety rules enforced here:
  * a Where is only touched when it has exactly one consumer and that consumer is MatMul/exMatMul
  * a Where is only touched when exactly one of its two data branches traces back to a Softmax
  * the source file is opened read-only and the result is written to a different path
  * after rewiring, nodes unreachable from the graph outputs are dropped (backward
    reachability, not "no consumer" -- the dead Equal/Not pair consumes each other)
  * the result is topologically checked before anything is written
"""
import argparse
import collections
import hashlib
import os

import onnx

SOFTMAX_OPS = {"Softmax", "exSoftmax13", "exSoftmax"}
MATMUL_OPS = {"MatMul", "exMatMul", "Gemm"}


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ancestors(prod, tensor, cap=400):
    """Op types reachable upstream of `tensor`, following every input."""
    seen, stack, out = set(), [tensor], set()
    while stack and len(out) < cap:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        p = prod.get(cur)
        if p is None:
            continue
        out.add(p.op_type)
        stack.extend(p.input)
    return out


def topo_check(nodes, seeds, graph_outputs):
    """Kahn topological sort. `seeds` must hold every name that needs no producer."""
    available = set(seeds)
    remaining = list(nodes)
    ordered = []
    progress = True
    while remaining and progress:
        progress = False
        nxt = []
        for n in remaining:
            if all((i in available) or (i == "") for i in n.input):
                ordered.append(n)
                available.update(n.output)
                progress = True
            else:
                nxt.append(n)
        remaining = nxt
    if remaining:
        return False, [n.name or n.op_type for n in remaining[:8]]
    return True, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    assert os.path.abspath(args.onnx) != os.path.abspath(args.out), \
        "refusing to overwrite the source graph"
    print("### SRC %s size=%d md5=%s" % (args.onnx, os.path.getsize(args.onnx), md5(args.onnx)),
          flush=True)

    m = onnx.load(args.onnx, load_external_data=True)
    g = m.graph
    before_nodes = len(g.node)
    before_hist = collections.Counter(n.op_type for n in g.node)

    prod = {}
    for n in g.node:
        for o in n.output:
            prod[o] = n
    cons = collections.defaultdict(list)
    for n in g.node:
        for i in n.input:
            cons[i].append(n)
    graph_out_names = {o.name for o in g.output}

    removed, rewire, skipped = [], {}, []
    for n in g.node:
        if n.op_type != "Where":
            continue
        out_consumers = cons[n.output[0]]
        if len(out_consumers) != 1 or out_consumers[0].op_type not in MATMUL_OPS:
            skipped.append((n.name, "consumer", [c.op_type for c in out_consumers]))
            continue
        if n.output[0] in graph_out_names:
            skipped.append((n.name, "graph_output", []))
            continue
        data_candidates = [t for t in (n.input[1], n.input[2])
                           if ancestors(prod, t) & SOFTMAX_OPS]
        if len(data_candidates) != 1:
            skipped.append((n.name, "softmax-branch-count=%d" % len(data_candidates), []))
            continue
        keep = data_candidates[0]
        drop = n.input[2] if keep == n.input[1] else n.input[1]
        rewire[n.output[0]] = keep
        removed.append({"where": n.name, "kept": keep, "dropped": drop,
                        "consumer": out_consumers[0].name or out_consumers[0].op_type})

    print("### Where nodes seen: %d   guards removable: %d   skipped: %d"
          % (sum(1 for n in g.node if n.op_type == "Where"), len(removed), len(skipped)),
          flush=True)
    print("### consumers of each guard Where, listed before any change:", flush=True)
    for r in removed[:8]:
        print("      where=%s  single consumer=%s  kept=%s  dropped=%s"
              % (r["where"], r["consumer"], r["kept"], r["dropped"]), flush=True)
    if len(removed) > 8:
        print("      ... %d more" % (len(removed) - 8), flush=True)
    for s in skipped[:6]:
        print("      SKIPPED %r" % (s,), flush=True)

    if not removed:
        print("### NO_GUARDS_FOUND: nothing to strip, refusing to write an identical graph",
              flush=True)
        raise SystemExit(2)
    if args.dry_run:
        print("### DRY_RUN, nothing written", flush=True)
        return

    for n in g.node:
        for i, t in enumerate(n.input):
            if t in rewire:
                n.input[i] = rewire[t]

    dropped_names = {r["where"] for r in removed}
    kept = [n for n in g.node if n.name not in dropped_names]

    # Backward reachability from the graph outputs: drops the dead guard branch
    # (Equal/Not consume each other, so "has no consumer" would never remove them).
    reachable, stack = set(), []
    for n in kept:
        if any(o in graph_out_names for o in n.output):
            stack.append(n)
    while stack:
        n = stack.pop()
        if id(n) in reachable:
            continue
        reachable.add(id(n))
        for i in n.input:
            p = prod.get(i)
            if p is not None:
                stack.append(p)
    after = [n for n in kept if id(n) in reachable]
    print("### DCE: %d -> %d nodes (%d dropped as unreachable)" % (len(kept), len(after), len(kept) - len(after)),
          flush=True)

    del g.node[:]
    g.node.extend(after)

    seeds = [i.name for i in g.input] + [t.name for t in g.initializer]
    ok, leftover = topo_check(g.node, seeds, graph_out_names)
    print("### TOPO_CHECK %s %s" % ("PASS" if ok else "FAIL", leftover or ""), flush=True)
    if not ok:
        raise SystemExit("topological order check failed; nothing written")

    after_hist = collections.Counter(n.op_type for n in g.node)
    print("### nodes before=%d after=%d" % (before_nodes, len(g.node)), flush=True)
    for k in sorted(set(before_hist) | set(after_hist)):
        if before_hist[k] != after_hist[k]:
            print("###   %-16s %d -> %d" % (k, before_hist[k], after_hist[k]), flush=True)

    onnx.checker.check_model(m)
    print("### ONNX_CHECKER PASS", flush=True)
    onnx.save(m, args.out)
    print("### OUT %s size=%d md5=%s" % (args.out, os.path.getsize(args.out), md5(args.out)),
          flush=True)
    print("### GUARD_STRIP_DONE", flush=True)


if __name__ == "__main__":
    main()
