#!/usr/bin/env python3
import argparse, hashlib, json, math, os, time
import statistics
from pathlib import Path

import torch

from kev.api import SystemOneRequest, to_record
from kev.checkpoint import LoadOptions
from kev.data import api_request
from kev.device import sync
from kev.predictors import LocalPredictor


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def record(case, reverse=False):
    options = list(case["options"])
    if reverse:
        options.reverse()
    criteria = {o["title"]: o.get("description", "") for o in options}
    return {
        "state": case["input"],
        "questions": {
            "command": {
                "type": "choice",
                "instructions": case["prompt"],
                "criteria": criteria,
            }
        },
    }


def score(predictor, request):
    """Run the serving input without adding gold labels to the model request."""
    started = time.perf_counter()
    api = api_request({"state": request["state"], "questions": request["questions"]})
    internal, _ = to_record(SystemOneRequest.model_validate(api))
    enc = predictor.model.encode(predictor.tok, internal, max_state=predictor.context["max_state"],
                                 max_branch=predictor.context["max_branch"], strict=True)
    sync(predictor.device)
    model_started = time.perf_counter()
    logits = predictor.model.forward(enc)
    sync(predictor.device)
    model_ms = 1000 * (time.perf_counter() - model_started)
    probs = torch.softmax(logits[0], -1).cpu().tolist()
    elapsed_ms = 1000 * (time.perf_counter() - started)
    titles = list(request["questions"]["command"]["criteria"])
    return {
        "probabilities": dict(zip(titles, probs)),
        "model_latency_ms": model_ms,
        "end_to_end_ms": elapsed_ms,
        "input_tokens": len(enc["ids"]),
    }


def finite_distribution(p):
    vals = list(p.values())
    return bool(vals) and all(math.isfinite(x) and x >= 0 for x in vals) and abs(sum(vals) - 1.0) < 1e-4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--repeat", type=int, default=50)
    args = ap.parse_args()

    cases = [json.loads(x) for x in open(args.cases, encoding="utf-8") if x.strip()]
    cases_sha = sha256(args.cases)
    options = LoadOptions(merge=True, fused=False, cuda_graphs=False)
    predictor = LocalPredictor(args.checkpoint, args.device, options)

    rows = []
    for case in cases:
        for reordered in (False, True):
            rec = record(case, reordered)
            started = time.perf_counter()
            error = None
            try:
                pred = score(predictor, rec)
                elapsed_ms = pred["model_latency_ms"]
                title_probs = pred["probabilities"]
                title_to_id = {o["title"]: o["id"] for o in case["options"]}
                probs = {title_to_id[k]: v for k, v in title_probs.items()}
                chosen = title_to_id[max(title_probs, key=title_probs.get)]
                valid = finite_distribution(probs)
                if not valid:
                    chosen = None
            except Exception as exc:
                elapsed_ms = (time.perf_counter() - started) * 1000
                probs, chosen, valid = {}, None, False
                error = repr(exc)
            rows.append({
                "id": case["id"],
                "language": case["language"],
                "group": case.get("group"),
                "category": case.get("category"),
                "reordered": reordered,
                "expected_id": case["expected_id"],
                "chosen_id": chosen,
                "correct": bool(valid and chosen == case["expected_id"]),
                "finite_distribution": valid,
                "probabilities": probs,
                "latency_ms": elapsed_ms,
                "end_to_end_ms": pred["end_to_end_ms"] if error is None else elapsed_ms,
                "input_tokens": pred["input_tokens"] if error is None else None,
                "error": error,
                "model_visible_input": rec,
            })

    timing_case = cases[0]
    timing_record = record(timing_case, False)
    warm = []
    for _ in range(args.warmup):
        warm.append(score(predictor, timing_record)["model_latency_ms"])
    samples = []
    end_to_end_samples = []
    input_tokens = None
    for _ in range(args.repeat):
        timed = score(predictor, timing_record)
        samples.append(timed["model_latency_ms"])
        end_to_end_samples.append(timed["end_to_end_ms"])
        input_tokens = timed["input_tokens"]
    ordered = sorted(samples)
    ordered_e2e = sorted(end_to_end_samples)
    p95 = ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]
    p95_e2e = ordered_e2e[min(len(ordered_e2e) - 1, math.ceil(0.95 * len(ordered_e2e)) - 1)]
    valid_perm = [
        (rows[i], rows[i + 1])
        for i in range(0, len(rows), 2)
        if rows[i]["error"] is None and rows[i + 1]["error"] is None
        and rows[i]["finite_distribution"] and rows[i + 1]["finite_distribution"]
        and rows[i]["chosen_id"] is not None and rows[i + 1]["chosen_id"] is not None
    ]
    by_language = {}
    for lang in sorted({r["language"] for r in rows}):
        lang_rows = [r for r in rows if r["language"] == lang]
        by_language[lang] = {
            "rows": len(lang_rows),
            "accuracy": sum(r["correct"] for r in lang_rows) / len(lang_rows),
            "accuracy_original": sum(r["correct"] for r in lang_rows if not r["reordered"]) / max(1, sum(not r["reordered"] for r in lang_rows)),
            "accuracy_reordered": sum(r["correct"] for r in lang_rows if r["reordered"]) / max(1, sum(r["reordered"] for r in lang_rows)),
            "finite_rate": sum(r["finite_distribution"] for r in lang_rows) / len(lang_rows),
            "errors": sum(bool(r["error"]) for r in lang_rows),
        }
    by_category = {}
    for category in sorted({r["category"] for r in rows}):
        category_rows = [r for r in rows if r["category"] == category]
        by_category[category] = {
            "rows": len(category_rows),
            "accuracy": sum(r["correct"] for r in category_rows) / len(category_rows),
            "accuracy_original": sum(r["correct"] for r in category_rows if not r["reordered"]) / max(1, sum(not r["reordered"] for r in category_rows)),
            "accuracy_reordered": sum(r["correct"] for r in category_rows if r["reordered"]) / max(1, sum(r["reordered"] for r in category_rows)),
            "errors": sum(bool(r["error"]) for r in category_rows),
        }
    action_rows = [r for r in rows if r["expected_id"] not in ("clarify", "reject")]
    safety_rows = [r for r in rows if r["expected_id"] in ("clarify", "reject")]
    result = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_head_sha256": sha256(Path(args.checkpoint) / "head.pt"),
        "checkpoint_adapter_sha256": sha256(Path(args.checkpoint) / "adapter_model.safetensors"),
        "checkpoint_tokenizer_sha256": sha256(Path(args.checkpoint) / "tokenizer.json"),
        "cases": os.path.abspath(args.cases),
        "cases_sha256": cases_sha,
        "device": args.device,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "n_cases": len(cases),
        "n_quality_rows": len(rows),
        "accuracy": sum(r["correct"] for r in rows) / len(rows),
        "accuracy_original": sum(r["correct"] for r in rows if not r["reordered"]) / len(cases),
        "accuracy_reordered": sum(r["correct"] for r in rows if r["reordered"]) / len(cases),
        "finite_rate": sum(r["finite_distribution"] for r in rows) / len(rows),
        "errors": sum(bool(r["error"]) for r in rows),
        "permutation_pairs": len(cases),
        "permutation_valid_pairs": len(valid_perm),
        "permutation_same_choice": (
            sum(a["chosen_id"] == b["chosen_id"] for a, b in valid_perm) / len(valid_perm)
            if valid_perm else None
        ),
        "accuracy_by_language": by_language,
        "accuracy_by_category": by_category,
        "gold_partition": {
            "action": {"rows": len(action_rows), "accuracy": sum(r["correct"] for r in action_rows) / max(1, len(action_rows))},
            "clarify_or_reject": {"rows": len(safety_rows), "accuracy": sum(r["correct"] for r in safety_rows) / max(1, len(safety_rows))},
        },
        "timing": {
            "case_id": timing_case["id"],
            "warmup": args.warmup,
            "n": args.repeat,
            "input_tokens": input_tokens,
            "cache_mode": "disabled; each request encodes and forwards independently",
            "model_latency_ms": samples,
            "end_to_end_ms": end_to_end_samples,
            "p50_ms": statistics.median(samples),
            "p95_ms": p95,
            "max_ms": ordered[-1],
            "end_to_end_p50_ms": statistics.median(end_to_end_samples),
            "end_to_end_p95_ms": p95_e2e,
            "end_to_end_max_ms": ordered_e2e[-1],
        },
        "cache_mode": "no cross-request prefix cache; predictor model forward recomputes each request",
        "model_visible_input_contract": "state,input; questions.command.type,instructions,criteria localized title->description; no gold label/source",
        "provenance": {
            "kev_source_commit": "6b719c3c3f367295f6ef336f4f751cf5ff970abc",
            "kev_model_revision": "9a45d25eb2ab761841196625383fa1dff0e56c1e",
            "qwen_base_revision": "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68",
            "adapter_model_sha256": sha256(Path(args.checkpoint) / "adapter_model.safetensors"),
            "runtime_head_sha256": sha256(Path(args.checkpoint) / "head.pt"),
            "cases_sha256": cases_sha,
            "eval_script_sha256": sha256(__file__),
        },
        "rows": rows,
    }
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
