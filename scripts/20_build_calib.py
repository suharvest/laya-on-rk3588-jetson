"""Build the RKNN int8 calibration set for the fixed seq=90 / markers=16 bucket.

Sequences come from laya's own `build_sequence`, so the token mix is the one the model
actually sees: a [CLS]-led head of instructions plus one [MASK]-led span per option, then as
much of the serialized state as the bucket leaves room for. Sampling uniform random token ids
would calibrate the activations against inputs the model never produces.

Output layout -- one directory per sample, npy files named after the ONNX inputs:

    <out>/calib/sample_0000/{input_ids,attention_mask,marker_pos,marker_mask,qtype}.npy
    <out>/calib.txt            one sample directory per line (toolkit multi-input dataset form)
    <out>/heldout/sample_0000/...
    <out>/manifest.json        per-sample metadata for both splits
"""
import argparse
import json
import os
import random

import numpy as np
import torch

from laya.agent import Agent
from laya.common import QTYPES, build_sequence, render_options

INPUT_NAMES = ["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"]

# Multilingual on purpose: the checkpoint is laya-multilingual, so the state text that reaches
# the tokenizer in production is not English-only.
SENTENCES = [
    "The order was placed on the customer portal yesterday evening and the shipping address was confirmed by the buyer.",
    "包装箱在运输途中出现了轻微的挤压变形，但内部商品完好无损。",
    "El cliente solicito una factura con el numero de identificacion fiscal actualizado antes del cierre del mes.",
    "退款申请已经进入审核队列，通常需要三个工作日才能完成处理。",
    "The warehouse supervisor noted that the pallet did not match the manifest for the second time this week.",
    "配送人员反馈该地址的门禁系统在夜间无法正常响应呼叫。",
    "A partial refund was issued for the damaged item and the remainder of the order is still in transit.",
    "客户希望将原先选择的蓝色款式更换为黑色款式，并且询问是否会产生额外费用。",
    "The support ticket was escalated after the first response failed to resolve the reported billing discrepancy.",
    "该批次产品的序列号与质检记录之间存在不一致，需要仓库重新核对。",
    "Le client demande une confirmation ecrite avant que le remboursement ne soit traite.",
    "The automated reminder was sent twice because the first message was caught by the spam filter.",
    "发票抬头需要修改为公司全称，并且补充纳税人识别号信息。",
    "The replacement unit shipped from a different distribution centre, so the tracking number changed.",
    "监控记录显示包裹在分拣中心停留了超过四十八小时未更新状态。",
    "Customer reports that the appliance stops heating after roughly ten minutes of continuous operation.",
    "售后工程师已经上门检查过一次，但问题在两周之后再次出现。",
]

INSTRUCTIONS = {
    "choice": ["Pick the single best matching option.", "Choose the option that best describes the request.",
               "Select exactly one of the listed categories.", "Which of these applies to the request?"],
    "score": ["Rate how severe the reported problem is.", "Score the urgency of the request.",
              "Assign a level to the current resolution stage.", "How advanced is the request right now?"],
    "noul": ["Is the statement below true for this request?", "Decide whether the claim holds.",
             "Answer the yes or no question about the state.", "Does the following condition apply?"],
}

# A 16-option question only fits a 90-token bucket if the head and the option spans are short:
# the head is capped at head_max_len (192) independently of the bucket, and everything past
# token 90 is cut, which silently drops the trailing option markers. So long questions are
# generated with a terse instruction and label-only options, which is also the form the
# renderer supports (`str(k) if v is None or v == ""`).
SHORT_INSTRUCTIONS = {
    "choice": ["Pick one.", "Which one?", "Choose."],
    "score": ["Rate it.", "How bad?", "Score."],
    "noul": ["True?", "Does it hold?", "Yes or no."],
}
SHORT_DESCS = ["yes", "no", "bug", "help", "pay", "ship", "fix", "new", "old", "lost", "late", "ok"]

LABELS = ["refund", "replacement", "repair", "escalate", "inform", "reject", "hold", "inspect",
          "reship", "credit", "waive", "callback", "priority", "standard", "return", "close",
          "transfer", "verify", "schedule", "cancel"]

DESCS = ["the customer asked for money back", "a new unit is required", "the item can be fixed on site",
         "the case exceeded the first response window", "no action is needed yet",
         "the request is outside the return period", "waiting on the customer to reply",
         "a technician has to look at the photos", "the parcel has to go out again",
         "the account has a credit balance", "the shipping fee should be removed",
         "the customer wants a call back today", "the ticket is marked urgent",
         "the ticket can follow the normal queue", "the goods have to come back to the warehouse",
         "the ticket can be closed once confirmed", "another team owns this request",
         "the identity of the caller is unconfirmed", "a visit has to be booked",
         "the customer changed their mind"]

SCORE_DESCS = ["not blocking anything", "slows the customer down", "needs attention this week",
               "needs attention today", "the customer is blocked", "the customer is threatening to churn"]


def make_state(rng, n_sent):
    return " ".join(rng.sample(SENTENCES, min(n_sent, len(SENTENCES))))


def make_question(rng, qtype, n_opts, compact=False):
    ins = rng.choice(SHORT_INSTRUCTIONS[qtype] if compact else INSTRUCTIONS[qtype])
    if qtype == "choice":
        labels = rng.sample(LABELS, n_opts)
        crit = {}
        for lab in labels:
            if compact:
                crit[lab] = "" if rng.random() < 0.5 else rng.choice(SHORT_DESCS)
            else:
                crit[lab] = rng.choice(DESCS) if rng.random() < 0.75 else ""
        return {"t": "choice", "ins": ins, "crit": crit}
    if qtype == "score":
        pool = SHORT_DESCS if compact else SCORE_DESCS
        return {"t": "score", "ins": ins, "crit": [rng.choice(pool) for _ in range(n_opts)]}
    # noul is always a two-option question: render_options emits exactly [false, true].
    pool = SHORT_DESCS if compact else DESCS
    return {"t": "noul", "ins": ins, "crit": {"false": rng.choice(pool), "true": rng.choice(pool)}}


def build_split(agent, rng, n, seq_len, num_markers, meta_out):
    tok = agent.tok
    kept, dropped = [], 0
    attempts = 0
    while len(kept) < n and attempts < n * 40:
        attempts += 1
        qtype = rng.choice(["choice", "score", "noul"])
        if qtype == "noul":
            n_opts = 2
        else:
            n_opts = rng.choice([1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16])
        n_sent = rng.choice([1, 2, 2, 3, 4, 5, 6, 8, 10, 12])
        state = make_state(rng, n_sent)
        q = make_question(rng, qtype, n_opts, compact=(n_opts >= 8))
        ids, markers = build_sequence(tok, state, q, seq_len, 192)
        n_options = len(render_options(q))
        # The runtime rejects a question whose options did not all survive the head budget
        # (agent.py raises when len(markers) != len(render_options(q))). Calibrating on rows the
        # serving path refuses would shift the activation ranges toward inputs it never runs.
        if len(markers) != n_options or not markers:
            dropped += 1
            continue
        if len(markers) > num_markers:
            dropped += 1
            continue

        L, K = seq_len, num_markers
        used = min(len(ids), L)
        input_ids = np.zeros((1, L), dtype=np.int64)
        attention_mask = np.zeros((1, L), dtype=np.int64)
        input_ids[0, :used] = np.asarray(ids[:used], dtype=np.int64)
        attention_mask[0, :used] = 1
        marker_pos = np.zeros((1, K), dtype=np.int64)
        marker_mask = np.zeros((1, K), dtype=np.bool_)
        k = len(markers)
        marker_pos[0, :k] = np.asarray(markers, dtype=np.int64)
        marker_mask[0, :k] = True
        qtype_arr = np.asarray([QTYPES[q["t"]]], dtype=np.int64)

        kept.append(({"input_ids": input_ids, "attention_mask": attention_mask,
                      "marker_pos": marker_pos, "marker_mask": marker_mask, "qtype": qtype_arr},
                     {"qtype": q["t"], "n_options": n_options, "n_markers": k,
                      "seq_len": int(used), "truncated": bool(len(ids) > L),
                      "state_sentences": n_sent, "instructions": q["ins"],
                      "options": render_options(q)}))
    return kept, dropped, attempts


def write_split(out_dir, items):
    os.makedirs(out_dir, exist_ok=True)
    dirs, meta = [], []
    for i, (arrays, m) in enumerate(items):
        d = os.path.join(out_dir, "sample_%04d" % i)
        os.makedirs(d, exist_ok=True)
        for name in INPUT_NAMES:
            np.save(os.path.join(d, name + ".npy"), arrays[name])
        dirs.append(d)
        meta.append(m)
    return dirs, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/harve/laya-rknn/models/laya-multilingual")
    ap.add_argument("--out", default="/home/harve/laya-rknn/calib_s90")
    ap.add_argument("--seq-len", type=int, default=90)
    ap.add_argument("--num-markers", type=int, default=16)
    ap.add_argument("--n-calib", type=int, default=128)
    ap.add_argument("--n-heldout", type=int, default=16)
    ap.add_argument("--calib-seed", type=int, default=20260929)
    ap.add_argument("--heldout-seed", type=int, default=771)
    args = ap.parse_args()

    print("Loading laya Agent (compile=False, device=cpu) from %s" % args.model)
    agent = Agent(args.model, compile=False, device="cpu")
    print("golden ONNX input names expected:", INPUT_NAMES)

    manifest = {"seq_len": args.seq_len, "num_markers": args.num_markers,
                "input_names": INPUT_NAMES, "splits": {}}

    for split, n, seed in (("calib", args.n_calib, args.calib_seed),
                           ("heldout", args.n_heldout, args.heldout_seed)):
        rng = random.Random(seed)
        items, dropped, attempts = build_split(agent, rng, n, args.seq_len, args.num_markers, None)
        if len(items) < n:
            raise SystemExit("only built %d/%d %s samples in %d attempts" % (len(items), n, split, attempts))
        dirs, meta = write_split(os.path.join(args.out, split), items)
        if split == "calib":
            with open(os.path.join(args.out, "calib.txt"), "w") as f:
                f.write("\n".join(dirs) + "\n")
        manifest["splits"][split] = {
            "n": len(items), "seed": seed, "dropped": dropped, "attempts": attempts,
            "dir": os.path.join(args.out, split),
            "qtype_counts": {t: sum(1 for m in meta if m["qtype"] == t)
                             for t in ("choice", "score", "noul")},
            "n_marker_hist": {str(k): sum(1 for m in meta if m["n_markers"] == k)
                              for k in sorted({m["n_markers"] for m in meta})},
            "seq_len_hist": {str(k): sum(1 for m in meta if m["seq_len"] == k)
                             for k in sorted({m["seq_len"] for m in meta})},
            "truncated": sum(1 for m in meta if m["truncated"]),
            "samples": meta,
        }
        print("### %s: %d samples, dropped=%d attempts=%d" % (split, len(items), dropped, attempts))
        print("    qtypes:", manifest["splits"][split]["qtype_counts"])
        print("    markers:", manifest["splits"][split]["n_marker_hist"])
        print("    seq_len:", manifest["splits"][split]["seq_len_hist"])
        print("    truncated:", manifest["splits"][split]["truncated"])

    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # The toolkit needs every input of every row; assert the shapes before the conversion
    # rather than after a build that takes minutes.
    for name in INPUT_NAMES:
        p = os.path.join(args.out, "calib", "sample_0000", name + ".npy")
        a = np.load(p)
        print("  calib sample signature %-15s shape=%-10s dtype=%s" % (name, a.shape, a.dtype))
    print("DONE")


if __name__ == "__main__":
    main()
