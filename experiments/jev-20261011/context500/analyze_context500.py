#!/usr/bin/env python3
"""Derive paired accuracy and timing slices from a completed context500 JSONL run."""
from __future__ import annotations
import argparse, json, math, statistics
from pathlib import Path

EXECUTION = {"clarify", "reject"}

def stats(values):
    if not values: return None
    xs = sorted(float(x) for x in values)
    return {"n": len(xs), "mean_ms": statistics.fmean(xs), "min_ms": xs[0], "p50_ms": statistics.median(xs), "p95_ms": xs[min(len(xs)-1, math.ceil(.95*len(xs))-1)], "max_ms": xs[-1]}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--dataset", required=True); ap.add_argument("--results", required=True); ap.add_argument("--out", required=True); args=ap.parse_args()
    dataset=[json.loads(x) for x in Path(args.dataset).read_text().splitlines() if x.strip()]
    result=[json.loads(x) for x in Path(args.results).read_text().splitlines() if x.strip()]
    by_key={(r["pass"],r["id"]):r for r in result}
    if len(dataset)!=120 or len(by_key)!=240: raise SystemExit(f"expected 120 dataset rows and 240 unique result rows; got {len(dataset)} and {len(by_key)}")
    def rows_for(field, value): return [r for r in result if r[field]==value]
    def accuracy(rs): return {"n":len(rs),"http_success":sum(r.get("status")==200 for r in rs),"correct":sum(bool(r.get("correct")) for r in rs),"accuracy":sum(bool(r.get("correct")) for r in rs)/len(rs) if rs else None}
    expected_by={(r["id"]):r["expected_id"] for r in dataset}
    false_action=[]
    for r in result:
        expected=expected_by[r["id"]]; selected=r.get("selected_option_id")
        if r.get("status")==200 and expected in EXECUTION and selected not in EXECUTION:
            false_action.append({"pass":r["pass"],"id":r["id"],"expected_id":expected,"selected_option_id":selected})
    paired=[]
    for row in dataset:
        a=by_key[("original",row["id"])]; b=by_key[("reverse",row["id"])]
        paired.append({"id":row["id"],"original_selected":a.get("selected_option_id"),"reverse_selected":b.get("selected_option_id"),"selected_id_changed":a.get("selected_option_id")!=b.get("selected_option_id"),"original_correct":bool(a.get("correct")),"reverse_correct":bool(b.get("correct")),"correctness_changed":bool(a.get("correct"))!=bool(b.get("correct"))})
    successful=[r for r in result if r.get("status")==200]
    complete=[r for r in successful if isinstance(r.get("server_timings"),dict) and all(isinstance(r["server_timings"].get(k),(int,float)) and math.isfinite(float(r["server_timings"][k])) for k in ("encoding_ms","inference_ms","total_ms"))]
    out={"dataset_cases":len(dataset),"scored_requests":len(result),"semantic_sample_denominator":120,"request_denominator":240,
         "overall":accuracy(result),"by_pass":{p:accuracy(rows_for("pass",p)) for p in ("original","reverse")},
         "by_language":{x:accuracy(rows_for("language",x)) for x in sorted({r["language"] for r in result})},
         "by_category":{x:accuracy(rows_for("category",x)) for x in sorted({r["category"] for r in result})},
         "false_action": {"definition":"HTTP 200 with expected clarify/reject and selected option outside {clarify,reject}; wrong selection of reject/clarify is excluded", "by_expected":{x:[r for r in false_action if r["expected_id"]==x] for x in ("clarify","reject")}, "count":len(false_action)},
         "paired_option_order": {"pairs":len(paired),"selected_id_changed_count":sum(x["selected_id_changed"] for x in paired),"selected_id_changed_rate":sum(x["selected_id_changed"] for x in paired)/len(paired),"correctness_changed_count":sum(x["correctness_changed"] for x in paired),"rows":paired},
         "performance": {"client_e2e_http200":stats([r["client_elapsed_ms"] for r in successful]),"server_encoding_complete":stats([r["server_timings"]["encoding_ms"] for r in complete]),"server_inference_complete":stats([r["server_timings"]["inference_ms"] for r in complete]),"server_total_complete":stats([r["server_timings"]["total_ms"] for r in complete]),"http200_count":len(successful),"timings_complete_count":len(complete),"timings_missing_count":len(successful)-len(complete)},
         "timing_scope_note":"Kev NativeBackend currently sets encoding_ms=0.0 as instrumentation; decision.py measures KevCodec.encode inside infer_request together with RKLLM hidden inference and FP32 head, so encoding_ms must not be reported as tokenizer wall time or as a measured zero-cost claim.",
         "method_note":"120 semantic cases are evaluated twice (original/reverse option order), not 240 independent semantic samples; five warmups are excluded by the runner."}
    Path(args.out).write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"out":args.out,"false_action_count":len(false_action),"paired_selected_change_count":out["paired_option_order"]["selected_id_changed_count"],"http200":len(successful),"timings_complete":len(complete)},ensure_ascii=False))
if __name__=="__main__": main()
