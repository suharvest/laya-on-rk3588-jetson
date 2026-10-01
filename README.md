<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License: Apache-2.0"></a>
  <img src="https://img.shields.io/badge/Platform-RK3588%20%7C%20RK3576%20%7C%20Jetson%20Orin-orange.svg" alt="Platform">
  <img src="https://img.shields.io/badge/Model-mmBERT--base%20322M-2f80ed.svg" alt="Model">
  <img src="https://img.shields.io/badge/Precision-fp16%20%7C%20INT8%20evaluated-f97316.svg" alt="Precision">
  <img src="https://img.shields.io/badge/Context-512%20tokens-6c757d.svg" alt="Context">
</p>

<h1 align="center">laya-on-rk3588-jetson</h1>

<p align="center">
  Run <a href="https://github.com/NandhaKishorM/laya">laya</a> — the open, Jev-compatible System One
  decision model — on Rockchip NPU and NVIDIA Jetson, entirely on-device.<br>
  <strong>RK3588 509 ms · Jetson Orin NX 18.9 ms · 16/16 decision accuracy</strong>
</p>

<!-- TODO: Add a demo visual — a ~15s terminal capture showing:
     (1) python3 live_rank.py against the RK3588 .rknn, printing the 16-row ranking table
     (2) the same command against the Jetson .engine
     Suggested tool: asciinema + agg, or Kap for screen recording.
     Place at media/demo.gif and reference it here. -->

## What is this?

Conversion recipes, measured latency and a reproducible acceptance protocol for
[`convaiinnovations/laya-multilingual`](https://huggingface.co/convaiinnovations/laya-multilingual)
(mmBERT-base, 322M parameters) on two edge NPU families: **Rockchip RK3588** (RKNN) and
**NVIDIA Jetson Orin** (TensorRT).

laya is a non-autoregressive System One decision model: one forward pass returns typed
choices, scores and yes/no answers over a piece of text, in 100+ languages. It implements the
same typed-decision request interface as TypeSafe's Jev, so it is the open path for the same
product shape — and, like any 322M encoder, it needs to be converted and quantized before it
will run on an NPU.

This repository is what came out of doing that conversion for real: the exact graph surgery
required, the latency each platform delivers, **which precisions do not work and why**, and a
ranking-based acceptance test that catches accuracy collapse which aggregate metrics miss.

## Performance

Fixed buckets, `seq = 512`. `top1` is the argmax over the model's live candidate classes on 16
held-out samples; `order_exact` is the full ranking of those classes. Reference is ONNX Runtime
fp32.

| Platform | Artifact | p50 | p95 | min | top1 | order_exact |
|---|---|---|---|---|---|---|
| **RK3588** (Radxa ROCK 5T) | `oldship.s512.m16.fp16.nanguard.sharedmask.rk3588.rknn` | **509 ms** | ≤598 ms | 493.7 ms | **16/16** | 15/16 |
| **Jetson Orin NX** | `laya.s512.fp16.engine` | **18.9 ms** | 24.8 ms | — | **16/16** | 16/16 |
| Jetson Orin Nano | same engine | 22.0 ms | 22.9 ms | — | **16/16** | 16/16 |

The two platforms differ by **27×**. Both match the fp32 reference on top1 for all 16 samples;
the RKNN model differs on the full ordering of one. Neither is bit-identical to the fp32
reference — fp16 output diverges from fp32 at the third decimal while preserving the ordering.

RK3588 p50 is only reproducible with the CPU governor pinned to `performance` — see
[`docs/measurement.md`](docs/measurement.md).

## What you get

- **Working conversion chain** for both platforms, including the ONNX graph surgery laya
  specifically requires. The mask construction in laya's export breaks RKNN accuracy unless two
  independent defects are fixed first; both are documented and scripted.
- **Measured numbers, not estimates** — latency, accuracy and memory, per platform, with the
  measurement protocol that produced them.
- **A quantized path that actually engages the INT8 tensor cores** (247/259 layers on Orin,
  `i8i8→i32`), and the reason its accuracy is still unusable.
- **Negative results with mechanisms**: six quantization attempts across three platforms, each
  failing for a different, identified reason. The ceiling is the quantization scheme, not the
  silicon.
- **The model's own limits**, measured separately from the deployment: 8k context does not work,
  and it does not work in fp32 either.
- **A ranking-based acceptance protocol** that would have caught two failures that `rel_l2`
  and layer tables both reported as passing.

## Quick start

Run the acceptance test against a converted artifact. This is the check every number in this
repository was gated on, and it is the check to run against us.

```bash
git clone https://github.com/suharvest/laya-on-rk3588-jetson
cd laya-on-rk3588-jetson
```

**1. On the target device** — run the converted model, dump what it decided:

```bash
python3 scripts/47_device_dump.py \
    --rknn   /path/to/oldship.s512.m16.fp16.nanguard.sharedmask.rk3588.rknn \
    --name   prod \
    --heldout calib/heldout \
    --out    dump.prod.json \
    --core-mask 0_1_2
```

**2. On any host** — score that dump against the fp32 reference:

```bash
python3 scripts/live_rank.py \
    --ref    golden/ortdump.s512.json \
    --dump   prod=dump.prod.json \
    --heldout calib/heldout \
    --out    rank.prod.json
```

The output is a per-sample table of the **live** candidate classes in ranked order, plus a
Kendall statistic. `--dump` is repeatable: pass several to compare models side by side in one
table.

Converted artifacts and the ONNX sources are published on Hugging Face:

| Artifact | Where |
|---|---|
| `.rknn`, `.engine`, both source `.onnx` files | [`harvestsu/laya-multilingual-rknn-tensorrt`](https://huggingface.co/harvestsu/laya-multilingual-rknn-tensorrt) |
| calibration + held-out sets, ORT reference | the same repository, under `calib/` and `golden/` (also committed here) |

Sizes and md5s for everything: [`ARTIFACTS.md`](ARTIFACTS.md).

## Conversion pipeline

```
HuggingFace checkpoint — convaiinnovations/laya-multilingual
        │
        ├── Rockchip path ──────────────────────────────────────────────────
        │     ONNX export (fp32)          seq bucket fixed here; mandatory
        │       │
        │       ├─ NaN guard repair       RKNPU has no IsNaN kernel
        │       ├─ attention-guard strip  removes a construct RKNN mis-lowers
        │       └─ mask-sharing surgery   22 per-layer masks → one shared mask
        │       │
        │       └─► rknn-toolkit2  --target rk3588 --optimization-level 3
        │                                │
        │                                └─► .rknn   663 MB, fp16
        │
        └── Jetson path ────────────────────────────────────────────────────
              ONNX export (a separate export; the two platforms do not share
              an ONNX graph in this repository)
                │
                └─► trtexec --fp16 --memPoolSize=workspace:2048   (sm_87)
                                │
                                └─► .engine 651 MB, fp16
```

The mask-sharing step is both an accuracy fix and a 209 ms latency fix on RK3588 — the two
problems have the same origin but are independent dimensions. See
[`docs/rk3588.md`](docs/rk3588.md).

## Configuration

| Knob | Where | Value used here | Effect |
|---|---|---|---|
| `seq` bucket | ONNX export | 512 | Fixed at export; short inputs are billed at 512 |
| `target_platform` | rknn-toolkit2 | `rk3588` | A rk3588 model is hard-rejected by an RK3576 runtime — convert separately |
| `optimization_level` | rknn-toolkit2 | 3 | |
| `core_mask` | rknnlite | `0_1_2` | Accepts enum values only; bit arithmetic is rejected |
| CPU governor | host | `performance` | RK3588 p50 is not reproducible without it |

## Documentation

| Doc | Contents |
|---|---|
| [`docs/rk3588.md`](docs/rk3588.md) | Rockchip path: the `H×W ≤ 8192` surface limit, graph surgery, conversion, results |
| [`docs/jetson.md`](docs/jetson.md) | TensorRT path: engine build, CUDA Graph, per-node bottleneck breakdown |
| [`docs/quantization.md`](docs/quantization.md) | Six attempts across three platforms, and where the ceiling actually is |
| [`docs/acceptance.md`](docs/acceptance.md) | The acceptance protocol, and three ways `rel_l2` and layer tables lied |
| [`docs/measurement.md`](docs/measurement.md) | Measurement discipline — how to get a number that reproduces |
| [`docs/context-length.md`](docs/context-length.md) | Long-context probe: 512 / 2048 / 8192, per language and per position |
| [`ARTIFACTS.md`](ARTIFACTS.md) | Every artifact with size and md5, and the exact ONNX each was built from |
| [`MANIFEST.md`](MANIFEST.md) | Pinned dependency manifests and upstream revisions, for rebuilds |

## Contributing

Issues and PRs are welcome, particularly:

- conversions for other Rockchip targets (RK3576, RK3588S) or other NPUs
- quantization schemes that survive the outlier channels per-tensor INT8 cannot
- reproductions or refutations of any number here — the acceptance protocol is in
  [`docs/acceptance.md`](docs/acceptance.md) and is meant to be run against us

## Acknowledgements

Built on [`NandhaKishorM/laya`](https://github.com/NandhaKishorM/laya) and
[`convaiinnovations/laya-multilingual`](https://huggingface.co/convaiinnovations/laya-multilingual),
both Apache-2.0. Upstream revisions and attribution are in [`NOTICE`](NOTICE).

Toolchains: [`rknn-toolkit2`](https://github.com/airockchip/rknn-toolkit2) (2.3.2),
TensorRT 10.3.0, onnx-graphsurgeon, ONNX Runtime.

## License

[Apache License 2.0](LICENSE). Converted artifacts are derivative works of
`laya-multilingual` and carry the same license and attribution.

This project is not affiliated with TypeSafe AI, Inc. "Jev" is their product; the references
here are descriptive.
