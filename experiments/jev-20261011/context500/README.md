# Kev context-500 benchmark

This benchmark measures the Kev RKLLM command-selection path with a fixed assembled context of exactly 500 tokens. It extends each frozen semantic request with fixed, neutral synthetic industrial-document metadata sentences. The added text does not describe an action, device state, sensor state, or operating condition, and the original input, prompt meaning, candidates, expected label, language, and category remain unchanged. The extension is a length-control method; it is not a claim about accuracy on naturally occurring long tasks.

The corpus contains 120 frozen semantic cases: 60 English and 60 Chinese cases across direct, paraphrase, negation, ambiguous, and out-of-scope categories. Each case is sent twice, with the original option order and with the option order reversed, for 240 scored requests per device. Five warmup requests are sent before scoring and are excluded from the 240-request denominator. A warmup failure stops the run.

## Reproducible inputs

- [`context500-cases.jsonl`](context500-cases.jsonl) contains the request bodies, expected labels, order variants, exact token counts, and per-request SHA256 values. Expected fields are metadata and are not sent in the HTTP body.
- [`context500-metadata.json`](context500-metadata.json) records the frozen source corpus SHA, deployed tokenizer SHA, Kev codec source SHA, delimiter IDs, target length, and generator/runner hashes.
- [`tokenizer.fe000.json`](tokenizer.fe000.json) is the tokenizer copied from the deployed Radxa release and verified as SHA256 `fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927`.
- [`generate_context500.py`](generate_context500.py) calls the application `KevCodec.encode` for every generated request and rejects any mismatch with its independent token-count check.
- [`run_context500.py`](run_context500.py) runs sequential loopback HTTP requests, supports hash-checked resume, performs no automatic retry, and writes one raw JSONL record per scored request plus a generated summary.

The deployed image, model, tokenizer, runtime, and Hugging Face revision must be recorded with each device result. The release pin and artifact evidence are maintained in the corresponding release evidence directory; this benchmark does not download or replace deployment artifacts.

## Measurement scope

Client elapsed time is measured around each loopback HTTP request. Server timing fields are reported separately as `encoding_ms`, `inference_ms`, and `total_ms`. The summary reports `n`, mean, minimum, p50, p95, and maximum for client end-to-end timing and for each available server timing field. HTTP errors, timeouts, and responses missing any server timing field are listed separately; missing timing data is not treated as a zero.

Accuracy is reported over the same 240 scored requests used for performance: overall, by language, by category, and by option-order pass. The summary includes the complete error list and the `wrong_predictions` list for HTTP 200 responses whose selected option differs from the expected label. The five warmups are not part of either accuracy or performance denominators. The service clears its KV cache for each request and does not share request history; the runner sends requests serially.

Hardware results are added under `rk3576/` and `rk3588/` only after the device worker has produced raw per-request JSONL, generated summary output, and artifact/hash evidence. Until those files are present, this document describes the test method and scope, not a completed device result.
