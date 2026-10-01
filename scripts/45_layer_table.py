"""Layer-table statistics from an rknn-toolkit2 console log.

The console log (not the verbose file) holds the per-op table with the placement column:
    D RKNN: [22:43:06.085] 49   exSoftmax13        FLOAT16  CPU  (1,12,512,512) ... exSoftmax13:...
plus the 'will fallback to CPU, because ...' warnings that explain each CPU placement.
"""
import collections
import re
import sys

ROW = re.compile(r"\]\s+(\d+)\s+(\S+)\s+(\S+)\s+(NPU|CPU|GPU)\s+(\()")
FALLBACK = re.compile(r"fallback to CPU, because (.*?)(?:!|$)")


def main(path):
    rows = []
    fallbacks = collections.Counter()
    n_lines = 0
    with open(path, "r", errors="replace") as f:
        for line in f:
            n_lines += 1
            m = FALLBACK.search(line)
            if m:
                fallbacks[m.group(1).strip()] += 1
            m = ROW.search(line)
            if m:
                rows.append({"id": int(m.group(1)), "op": m.group(2),
                             "dtype": m.group(3), "target": m.group(4),
                             "line": line.strip()[-160:]})
    print("### log: %s  lines=%d  layer-table rows=%d" % (path, n_lines, len(rows)), flush=True)
    by_target = collections.Counter(r["target"] for r in rows)
    print("### placement: %s" % dict(by_target), flush=True)
    by_target_op = collections.defaultdict(collections.Counter)
    for r in rows:
        by_target_op[r["target"]][r["op"]] += 1
    for tgt in sorted(by_target_op):
        print("###   %s ops: %s" % (tgt, dict(by_target_op[tgt].most_common())), flush=True)
    print("### fallback warnings: %d  %s" % (sum(fallbacks.values()), dict(fallbacks)), flush=True)
    for op in ("Transpose", "exSoftmax13", "exSoftmax", "Softmax"):
        hit = [r for r in rows if r["op"] == op]
        if not hit:
            continue
        c = collections.Counter(r["target"] for r in hit)
        print("### op %-12s rows=%d placement=%s" % (op, len(hit), dict(c)), flush=True)
        for r in hit[:3]:
            print("      %s" % r["line"], flush=True)
        cpu = [r for r in hit if r["target"] != "NPU"]
        if cpu:
            print("      NON-NPU rows: %d  e.g. %s" % (len(cpu), cpu[0]["line"]), flush=True)
    cpu_rows = [r for r in rows if r["target"] != "NPU"]
    cpu_ops = collections.Counter(r["op"] for r in cpu_rows)
    print("### non-NPU rows total=%d ops=%s" % (len(cpu_rows), dict(cpu_ops.most_common())), flush=True)
    print("### LAYER_TABLE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
