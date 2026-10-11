#!/usr/bin/env python3
"""Build exact-500-token Kev requests from the frozen 120-case corpus."""
from __future__ import annotations
import hashlib, json, re
from pathlib import Path
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "experiments/jev-20261010/shared/jev-shared-cases-20261010.jsonl"
TOKENIZER = Path("/home/harvest/tmp/jev-20261010/kev/model/tokenizer.json")
OUT = Path(__file__).with_name("context500-cases.jsonl")
TARGET = 500
DELIMS = (248060, 248061, 248049, 248050, 248062)
MARKER_RE = re.compile(r"<\|([^>]+)\|>")

EN_SENTENCES = [
    "Synthetic benchmark context: document metadata only.",
    "The following line is a neutral synthetic appendix.",
    "This appendix records no device action or state.",
    "It exists only to extend the benchmark input.",
    "The text uses fixed industrial documentation wording.",
    "The benchmark keeps the original request unchanged.",
    "No command, status, or operating condition is asserted.",
    "This is synthetic length context for measurement only.",
]
ZH_SENTENCES = [
    "合成基准上下文：仅记录文档格式元信息。",
    "以下句子是中性的合成附录。",
    "附录不记录设备动作或状态。",
    "它仅用于扩展基准输入长度。",
    "文本采用固定的工业文档措辞。",
    "原始请求内容保持不变。",
    "本段不声明命令、状态或运行条件。",
    "这段合成文字仅用于长度测量。",
]


def sha(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def escape(value: str) -> str:
    return MARKER_RE.sub(lambda m: "<¦" + m.group(1) + "¦>", value)


def enc(tok: Tokenizer, value: str) -> list[int]:
    return tok.encode(escape(value), add_special_tokens=False).ids


def request_for(case: dict, prompt: str, reverse: bool = False) -> dict:
    opts = list(case["options"])
    if reverse:
        opts.reverse()
    return {
        "input_text": case["input"],
        "prompt": prompt,
        "options": [
            {"id": o["id"], "label": o["title"], "description": o.get("description", "")}
            for o in opts
        ],
    }


def assembled_tokens(tok: Tokenizer, request: dict) -> int:
    ids = [DELIMS[0], *enc(tok, request["input_text"]), DELIMS[1], *enc(tok, request["prompt"])]
    for option in request["options"]:
        label = option["label"]
        desc = option.get("description", "")
        text = label + (": " + desc if desc else "")
        ids.extend([DELIMS[2], *enc(tok, text), DELIMS[3]])
    ids.append(DELIMS[4])
    return len(ids)


def solve(tok: Tokenizer, case: dict) -> tuple[str, list[str], int]:
    sentences = EN_SENTENCES if case["language"] == "en" else ZH_SENTENCES
    baseline = request_for(case, case["prompt"])
    base_n = assembled_tokens(tok, baseline)
    header = sentences[0]
    prefix = case["prompt"] + "\n\n" + header
    prefix_req = request_for(case, prefix)
    prefix_n = assembled_tokens(tok, prefix_req)
    if prefix_n >= TARGET:
        raise RuntimeError(f"{case['id']}: baseline/header already reaches {prefix_n}")

    # Tokenizer BPE boundaries are stable across these newline-separated sentences.
    # Use actual incremental counts from this case, then verify the constructed request.
    increments = []
    current = prefix
    for sentence in sentences:
        trial = current + "\n" + sentence
        increments.append(assembled_tokens(tok, request_for(case, trial)) - assembled_tokens(tok, request_for(case, current)))
    remaining = TARGET - prefix_n
    dp: dict[int, list[int]] = {0: []}
    for total in range(remaining + 1):
        if total not in dp:
            continue
        for idx, inc in enumerate(increments):
            nxt = total + inc
            if nxt <= remaining and nxt not in dp:
                dp[nxt] = dp[total] + [idx]
    if remaining not in dp:
        raise RuntimeError(f"{case['id']}: no natural sentence combination reaches {TARGET}; baseline={base_n}, header={prefix_n}")
    indices = dp[remaining]
    filler = [header] + [sentences[i] for i in indices]
    prompt = case["prompt"] + "\n\n" + "\n".join(filler)
    original = request_for(case, prompt)
    reverse = request_for(case, prompt, reverse=True)
    actual = assembled_tokens(tok, original)
    rev_actual = assembled_tokens(tok, reverse)
    if actual != TARGET or rev_actual != TARGET:
        raise RuntimeError(f"{case['id']}: verification failed original={actual} reverse={rev_actual} target={TARGET}")
    return prompt, filler, base_n


def main() -> None:
    tok = Tokenizer.from_file(str(TOKENIZER)); tok.no_truncation(); tok.no_padding()
    cases = [json.loads(line) for line in SOURCE.read_text().splitlines() if line.strip()]
    if len(cases) != 120:
        raise RuntimeError(f"expected 120 frozen cases, got {len(cases)}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as handle:
        for case in cases:
            prompt, filler, baseline_tokens = solve(tok, case)
            baseline = request_for(case, case["prompt"])
            request = request_for(case, prompt)
            reverse = request_for(case, prompt, reverse=True)
            row = {
                "id": case["id"], "language": case["language"], "category": case["category"],
                "group": case["group"], "expected_id": case["expected_id"],
                "baseline_request": baseline, "request_500": request,
                "baseline_tokens": baseline_tokens, "assembled_tokens": assembled_tokens(tok, request),
                "reverse_assembled_tokens": assembled_tokens(tok, reverse),
                "filler_sentences": filler,
                "filler_source": "fixed natural industrial-document metadata sentences in this generator; synthetic length extension only",
                "filler_method": "append to original prompt after two newlines; preserve input/options/expected label; verify KevCodec delimiter assembly",
                "baseline_request_sha256": sha(baseline), "request_500_sha256": sha(request),
                "reverse_request_500": reverse, "reverse_request_500_sha256": sha(reverse),
                "source_case_sha256": sha(case),
            }
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(OUT), "cases": len(cases), "target_assembled_tokens": TARGET,
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "tokenizer_sha256": hashlib.sha256(TOKENIZER.read_bytes()).hexdigest(),
        "dataset_sha256": hashlib.sha256(OUT.read_bytes()).hexdigest(),
    }, indent=2))

if __name__ == "__main__":
    main()
