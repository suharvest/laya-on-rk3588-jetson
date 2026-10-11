# Kev RK3588 RKLLM 8K evidence

- Engine: kev-0.8b-w8a8-rk3588-context8192-shortcalib.rkllm, SHA-256 f53070dd831deedf137d7acf639553a8920c1e391cc6da0738f82a357d3a161d.
- Runtime: RKLLM 1.3.1, RK3588, W8A8, 3 NPU cores; runtime reported max_context_limit: 8192 and init_ret=0.
- Input order: official state -> q/instruction -> options -> decide; frozen quality input excludes gold labels.
- Hidden output: every tested length returned all input-token rows with dimension 1024; no overflow through 8022 tokens.

## Capacity/performance

Board run used clear-cache per request. Short sequences used warmup 20 and 50 samples; longer sequences used warmup 2/sample 10, and 8022 used warmup 1/sample 5. Values are model run p50/p95 and full decision end-to-end p50/p95 in milliseconds:

| input tokens | run p50/p95 ms | e2e p50/p95 ms |
|---:|---:|---:|
| 118 | 831.966 / 870.606 | 840.142 / 878.989 |
| 144 | 1167.497 / 1213.002 | 1177.285 / 1225.894 |
| 534 | 3793.065 / 5145.880 | 3800.632 / 5154.633 |
| 1028 | 11811.523 / 12268.086 | 11818.415 / 12276.480 |
| 2068 | 24001.093 / 24309.724 | 24009.297 / 24316.615 |
| 4096 | 49559.075 / 50567.833 | 49566.858 / 50574.843 |
| 8022 | 111206.268 / 119427.397 | 111216.649 / 119445.738 |

These timings are this RK3588 board run, with the recorded shared load and sample counts; they are not a claim for 8K industrial throughput.

## Frozen quality

The 240-row run is split into independent 120-row original and 120-row reordered-option sets. Raw output is quality-8k-raw.log; derived aggregation is quality-8k-summary.json.

- Original: 100/120 (83.33%), Wilson 95% CI 75.65–88.94%; errors 0.
- Reordered: 100/120 (83.33%), Wilson 95% CI 75.65–88.94%; errors 0.
- EN: original 50/60 (83.33%), reordered 51/60 (85.00%). ZH: original 50/60 (83.33%), reordered 49/60 (81.67%).
- Gold gate: original clarify/reject 11/40 misclassified as action; reordered 9/40. Direct-action cases sent to clarify/reject: 1/80 in each split.
- By category, direct was 40/40 in both splits; ambiguous was 7/20 original and 5/20 reordered; OOS was 14/20 and 16/20; negation was 19/20 and 20/20; paraphrase was 20/20 and 19/20.

This is a synthetic 60 semantic-pair benchmark for conversion validation, not industrial acceptance. The engine used the successful short calibration export; the earlier long-calibration export attempts failed during WSL conversion and do not establish a causal OOM explanation.

## Reproducibility hashes

- quality raw: 2c284a75ca180a23ff186367bc3e5c4994e0fdfbe261d941217c069a7b6e057a
- quality summary: 784ff50855b3e8865d5ec088ab266c92ff2f914b99cf03ae09b190904035150e
- runtime binary: 4120ee672a0703b749f086c1cc1e12f52509c1beb45de8e1f73cf50a84c70e48
- runtime source: 3da5fe089b11d6048a8c3044b331d52cf4500bb8a5ec1e80530a2686bb7c1d421
- quality manifest: 9f177beb7da6cef7fac10fa50c2ee4d736aa2c29beaec59557f61d4b46eb8004
- frozen source IDs manifest: a647c769375d9282cc14cb9d03a5a0d10df50c1a30bad4ebcc01c35966d32f86

## Reproduction artifacts

`timing-summary.json` contains every per-sample clear, RKLLM run, FP32 head, and end-to-end timing for all seven token lengths, including warmup/sample counts and full-hidden checks. `long-input-manifest.json` binds the exact generated token IDs and option/decide indexes. `run_context8k_long.sh`, `run_context8k_timing.sh`, `runtime_context8k.cpp`, and `runtime_quality8k.cpp` are the self-written runner/source snapshots; their SHA-256 values are in `artifact-sha256.txt`. `official-all-ids-manifest.json`, `quality_manifest.tsv`, and `ids/` preserve the frozen input metadata and token IDs used for the board run. `provenance.json` binds the engine, source checkpoint, calibration, SDK, manifests, and runner hashes.
