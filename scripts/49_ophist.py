"""Parse the RKNN conversion-time layer tables out of two console logs and compare them.

The table rows look like:
D RKNN: [22:35:42.898] 16   Conv               FLOAT16  NPU    (1,768,1,512),(2304,768,1,1)             (1,2304,1,512)         282603/1769472/1769472   4224         Conv:/encoder/layers.0/attn/Wqkv/MatMul#2

Columns: ID OpType DataType Target InputShape OutputShape Cycles(DDR/NPU/Total) RW(KB) FullName
"""
import collections
import json
import re
import sys

ROW = re.compile(
    r"^D RKNN: \[(?P<ts>[0-9:.]+)\]\s+"
    r"(?P<id>\d+)\s+"
    r"(?P<op>\S+)\s+"
    r"(?P<dtype>\S+)\s+"
    r"(?P<target>NPU|CPU|GPU)\s+"
    r"(?P<in>\S+)\s+"
    r"(?P<out>\S+)\s+"
    r"(?P<ddr>\d+)/(?P<npu>\d+)/(?P<total>\d+)\s+"
    r"(?P<rw>\d+)\s+"
    r"(?P<name>\S+)\s*$"
)

HDR = re.compile(r"^D RKNN: \[[0-9:.]+\] ID\s+OpType\s+DataType\s+Target")


def parse(path):
    rows = []
    bad = 0
    with open(path, "r", errors="replace") as f:
        for line in f:
            if not line.startswith("D RKNN: ["):
                continue
            if HDR.match(line):
                continue
            m = ROW.match(line.rstrip("\n"))
            if m:
                d = m.groupdict()
                rows.append({
                    "id": int(d["id"]), "op": d["op"], "dtype": d["dtype"],
                    "target": d["target"], "in": d["in"], "out": d["out"],
                    "ddr": int(d["ddr"]), "npu": int(d["npu"]),
                    "total": int(d["total"]), "rw": int(d["rw"]), "name": d["name"],
                })
            else:
                bad += 1
    return rows, bad


def shapes_of(r):
    """All (shape-string) tensors this row mentions, keyed by kind."""
    return {"in": r["in"].split("),("), "out": r["out"].split("),(")}


def dims(shape_str):
    s = shape_str.strip().strip("()")
    if s in ("\\", "", "..."):
        return None
    try:
        return [int(x) for x in s.split(",") if x.strip() != ""]
    except ValueError:
        return None


def sq_pairs(shape_str, s):
    """Return the set of axis pairs (i,j) both equal to S, treating '...' as wildcard.

    A shape like (1,1,512,512) has the pair (2,3). (1,768,1,512) has none.
    """
    for tok in shape_str.split(","):
        t = tok.strip()
        if t == "...":
            continue
    ds = dims(shape_str)
    if ds is None:
        return []
    return [(i, j) for i in range(len(ds)) for j in range(i + 1, len(ds))
            if ds[i] == s and ds[j] == s]


def summarise(rows, label, s):
    hist = collections.Counter(r["op"] for r in rows)
    tot = collections.Counter()
    npu_cyc = collections.Counter()
    rw_sum = collections.Counter()
    tgt = collections.Counter()
    ss_rows = []
    for r in rows:
        tot[r["op"]] += 1
        npu_cyc[r["op"]] += r["total"]
        rw_sum[r["op"]] += r["rw"]
        tgt[(r["op"], r["target"])] += 1
        hit = bool(sq_pairs(r["in"], s)) or bool(sq_pairs(r["out"], s))
        if hit:
            ss_rows.append(r)
    return {"label": label, "n_rows": len(rows), "hist": hist, "cycles": npu_cyc,
            "rw": rw_sum, "target": tgt, "ss_rows": ss_rows}


def main():
    a_log, a_label, b_log, b_label, s = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
    A, badA = parse(a_log)
    B, badB = parse(b_log)
    print("PARSE %s rows=%d unparsed=%d" % (a_label, len(A), badA))
    print("PARSE %s rows=%d unparsed=%d" % (b_label, len(B), badB))

    sa = summarise(A, a_label, s)
    sb = summarise(B, b_label, s)

    print("\n===== OP HISTOGRAM =====")
    keys = sorted(set(sa["hist"]) | set(sb["hist"]))
    print("%-22s %10s %10s %10s" % ("OpType", a_label, b_label, "delta"))
    for k in keys:
        x, y = sa["hist"][k], sb["hist"][k]
        mark = "  <<<" if x != y else ""
        print("%-22s %10d %10d %10d%s" % (k, x, y, x - y, mark))

    print("\n===== SUM Cycles(Total) BY OP =====")
    print("%-22s %18s %18s %18s" % ("OpType", a_label, b_label, "delta"))
    for k in keys:
        x, y = sa["cycles"][k], sb["cycles"][k]
        print("%-22s %18d %18d %18d" % (k, x, y, x - y))

    print("\n===== SUM RW(KB) BY OP =====")
    for k in keys:
        x, y = sa["rw"][k], sb["rw"][k]
        print("%-22s %18d %18d %18d" % (k, x, y, x - y))

    print("\n===== TARGET SPLIT (op,target) =====")
    for k in sorted(set(sa["target"]) | set(sb["target"])):
        x, y = sa["target"][k], sb["target"][k]
        print("%-30s %8d %8d" % (str(k), x, y))

    print("\n===== S x S OPERATORS (S=%d) =====" % s)
    for tag, sm in ((a_label, sa), (b_label, sb)):
        print("\n--- %s : %d rows mentioning SxS ---" % (tag, len(sm["ss_rows"])))
        cc = collections.Counter(r["op"] for r in sm["ss_rows"])
        print("    by op:", dict(cc))
        tc = collections.Counter(r["target"] for r in sm["ss_rows"])
        print("    by target:", dict(tc))
        print("    sum RW(KB):", sum(r["rw"] for r in sm["ss_rows"]),
              " sum Cycles(Total):", sum(r["total"] for r in sm["ss_rows"]))
        for r in sm["ss_rows"]:
            print("    %4d %-16s %-8s %-4s in=%-34s out=%-24s cyc=%d/%d/%d rw=%d %s"
                  % (r["id"], r["op"], r["dtype"], r["target"], r["in"], r["out"],
                     r["ddr"], r["npu"], r["total"], r["rw"], r["name"]))

    json.dump({"a": {k: (dict(v) if isinstance(v, collections.Counter) else
                         (v if k != "ss_rows" else v)) for k, v in sa.items()},
               "b": {k: (dict(v) if isinstance(v, collections.Counter) else
                         (v if k != "ss_rows" else v)) for k, v in sb.items()}},
              open("/tmp/49_ophist.json", "w"), indent=1, default=str)
    print("\nWROTE /tmp/49_ophist.json")


main()
