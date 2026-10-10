# Kev-0.8B evaluation evidence

Evaluation date: 2026-10-10. This is a CPU correctness run on Spark; it is not a Jetson or CUDA performance result.

## Pinned inputs

- Kev source commit: `6b719c3c3f367295f6ef336f4f751cf5ff970abc`
- Kev model revision: `9a45d25eb2ab761841196625383fa1dff0e56c1e`
- Qwen3.5-0.8B-Base revision: `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`
- Frozen cases SHA256: `a1c18d3f86aa92eb11d06a5034ee1884de5b1228089b77648aeefa42fdd25a8a`
- Adapter SHA256: `9b908623acb162118575f4e7a94524f9c139c335be4bfb74d6cfceca01e1885a`
- Runtime head metadata SHA256: `4494415746482eeb427dbf69ea3e6fc2f1acdf65e7d494f055a160ef83657d6c`

The model-visible request contains only `state=input`, question type, instructions, and localized option title/description criteria. Gold fields are retained only in the result rows.

## CPU run

Command used `--device cpu --warmup 20 --repeat 50` with 120 frozen cases and one reordered copy of each case. Output JSON SHA256: `ab0d18c2d9b2dd5aefb38b713a48d39ba9106df3078112c0335008f26cafa598`.

- Accuracy: 0.8000 (192/240)
- Original order: 0.8083 (97/120)
- Reordered options: 0.7917 (95/120)
- Finite, normalized distributions: 240/240
- Errors: 0
- Same semantic choice after reorder: 0.9167 (110/120)
- English: 0.7750; Chinese: 0.8250
- Timing case input: 94 tokens; model-only p50/p95 1520.9/1612.3 ms; end-to-end p50/p95 1521.7/1613.2 ms

The CPU environment lacked `causal_conv1d` and `flash-linear-attention`, so the source reference PyTorch implementations were used. The timing is therefore a correctness-run baseline, not an edge-device estimate.

## CUDA smoke

Spark CUDA Torch was available (`torch 2.14.0+cu130`), but Kev initialization failed before inference with `CUDA_ERROR_OUT_OF_MEMORY` from `cuDevicePrimaryCtxRetain`. The basic Spark CUDA tensor smoke succeeded, but at the failed Kev load `torch.cuda.mem_get_info()` reported only about 13.0 GiB free; `nvidia-smi` also showed existing ASR/TTS processes. The exact allocation causing the context-retain failure is not isolated. No existing service was stopped. Failure log SHA256: `b3aeee2249af56a82db5c2efb797c5c511baf051fda6ce74af23c5f426583252`. Basic CUDA smoke evidence SHA256: `972e04ef4f8d8636dac5b4a8c15bb7150a7217a8ee0e72ddc3ed2e620ee29bbb`.

A CUDA/Jetson timing result remains pending a free GPU window. Spark SM121 artifacts must not be assumed portable to Jetson SM87.

## Offline derived metrics

`derived-metrics.json` is computed from the raw JSON and frozen cases without rerunning inference. Aggregation script SHA256: `7e5f1cca5bb73c4e88f1081f34c7370d56c2076f06667514848d68f654b9b44b`. It preserves separate original/reordered denominators and reports Wilson 95% intervals.

- Original 120: 97/120 correct, Wilson 95% CI 0.7288–0.8687. Clarify/reject precision 0.9355, recall 0.7250.
- Reordered 120: 95/120 correct, Wilson 95% CI 0.7105–0.8547. Clarify/reject precision 0.9375, recall 0.7500.
- Gold safety gate (40 clarify/reject cases): action errors 11/40 in original order and 10/40 after reordering; safety accuracy 19/40 and 17/40 respectively.
- Category accuracy, original/reordered: direct 1.00/1.00; paraphrase 0.95/0.95; negation 0.95/0.95; ambiguous 0.25/0.10; OOS 0.70/0.75.

The raw run was executed before later post-run additions to `kev_eval.py`; its embedded eval script hash is retained in the raw provenance.

## Orin NX preflight

Read-only preflight on `orin-nx` succeeded for CUDA fundamentals: JetPack driver CUDA 12.6, Torch `2.5.0a0+872d972e41.nv24.08`, compute capability `(8, 7)`, `torch.cuda.mem_get_info()` and a synchronized FP16 1024×1024 matmul both succeeded. No GPU processes were running during the check. The root filesystem had only 4.1 GiB free (99% used), so the ~1.75 GB base checkpoint was not transferred. Existing Python has `transformers 5.6.2`, `accelerate 1.13.0`, `tokenizers 0.22.2`, but no `peft`; NumPy 2.2.6 also emits an ABI warning with the vendor Torch build. Evidence is in `orin-nx-preflight.log` (SHA256 `683fb3c172e05ae76574649373d4782c5e848d624f05b863603c17a00f08925f`). Native Kev inference therefore remains untested on Orin NX; no package install, weight transfer, service stop, or TensorRT conversion was performed.


## Bounded Orin NX native attempt

The pinned runtime files were transferred once from Spark to Orin NX after a disk gate: 1,835,417,756 bytes total, with 4,056,633,344 bytes free beforehand and 2,220,756,992 bytes after transfer. Hashes matched the Spark artifacts. An isolated `uv venv --system-site-packages` reused the vendor Torch/Transformers and added only `peft==0.21.2` and then `datasets==5.1.0`, both with `--no-deps` from the TUNA index. The 5-case predictor smoke stopped before model loading: first `ModuleNotFoundError: No module named 'datasets'`; after the bounded dependency addition, importing `datasets` failed with `ModuleNotFoundError: No module named 'multiprocess'`. No further dependency installation was attempted. The evidence log is `nx-smoke-evidence.log` (SHA256 `b3af685692d5c43288dfb511e4b04f9650ff03d8331688b37e3cb538131a6fff`). Orin native model inference and timing remain unverified; the failure is an environment dependency closure issue, not a claim that Orin CUDA is unsupported.


### Final dependency-closure attempt

Package metadata showed that `datasets` requires `pyarrow`, `dill`, `multiprocess`, and `xxhash`; these were installed in one bounded operation together with venv-only `numpy==1.26.4`. Kev module import then reached the datasets/Hugging Face boundary but failed with `ImportError: cannot import name 'httpx' from huggingface_hub.utils`. The final environment versions were Torch `2.5.0a0+872d972e41.nv24.8`, Transformers `5.6.2`, huggingface-hub metadata `1.13.0`, PEFT `0.21.2`, datasets `5.1.0`, pyarrow `25.0.1`, dill `0.3.8`, multiprocess `0.70.16`, xxhash `3.5.0`, and NumPy `1.26.4`. The final import log is `nx-final-import.log` (SHA256 `69a292094c79a833dac8d326adb57ca33ccaaab2787f4ff3fa29fa88cdc64c5d`). No model load or 5-case inference was reached; no further package changes were made.


## Bekko comparison status

The Bekko browser dynamic INT8 ONNX route scored 79/120 on the original frozen order (English 40/60, Chinese 39/60), and 79/120 after option reordering. The fixed/native FP32 comparison scored 77/120 and is a separate, non-equivalent variant. The tested browser embedding shapes were prefix width 128 and document width 384.

The external WSL run compiled/exported the FP32 ONNX source model to an RKNN artifact: 291,453,711 bytes, SHA-256 `18570b7a1e61a256a15a0040e9819bb1472f0743c12eaab3c5a8c8641803025f`, elapsed 7:07.94, peak RSS 5.176 GB. Three WSL→Radxa/Spark transfers ended with SIGINT and no target file while both endpoints were online, so target runtime and formal NPU latency remain unverified. One native export parity sample measured `max_abs_logit_error=0.0540218353`; the 77/120 quality result is not full numeric parity. Model weights and bundled inference code have pending license declarations; no HF upload or public redistribution was performed.
