# Artifacts

Sizes and md5s for every artifact referenced in this repository, and the exact ONNX each
converted artifact was built from.

Large binaries (`.rknn`, `.engine`, `.onnx` — hundreds of MB each) are **not committed here**.
They are published separately; see the release/Hugging Face links in the [README](README.md).

---

## 1. Deliverables (seq = 512)

| Platform | File | Bytes | md5 |
|---|---|---|---|
| RK3588 (Radxa ROCK 5T) | `oldship.s512.m16.fp16.nanguard.sharedmask.rk3588.rknn` | 663,180,269 | `cff6c915b98b0400f888d55f5679fa69` |
| Jetson Orin NX / Nano | `laya.s512.fp16.engine` | 650,779,868 | `ae0688b17b297b1474a18a97e9e04c77` |

Each was verified identical at every location it was found:

- the `.rknn` on both the build host and the device — same md5
- the `.engine` on both Orin NX and Orin Nano (named `laya.s512.fp16.nxbuilt.engine` on Nano)
  — same md5

---

## 2. Source ONNX

**The two platforms do not share an ONNX graph.** They come from separate, differently-named
exports and undergo different graph surgery.

### 2.1 Rockchip path

Built from `oldship.s512.m16.nanguard.sharedmask.onnx`. Self-evidenced by the conversion log
(`31_convert_s512_sharedmask.console.log`):

```text
### label: s512-oldship-nanguard-sharedmask-fp16
### ONNX path: /mnt/f/laya-oldship/onnx/oldship.s512.m16.nanguard.sharedmask.onnx size: 1290265010
```

| Step | File | Bytes | md5 |
|---|---|---|---|
| 1 export | `oldship.s512.m16.fp32.onnx` | 1,290,284,930 | `9b63750ac25695ae4aa9f9870a36f359` |
| 2 NaN guard repair | `oldship.s512.m16.nanfix.onnx` | 1,290,290,684 | `6cca4461366158e3e29ad0795275a8c8` |
| 3 attention-guard strip | `oldship.s512.m16.nanguard.onnx` | 1,290,275,516 | `89c23f13597f098a43d859c66db64222` |
| 4 mask sharing | **`oldship.s512.m16.nanguard.sharedmask.onnx`** | **1,290,265,010** | **`5c68449182db6dd94587736bad27ccff`** |

Step 4 is the artifact that was converted.

### 2.2 Jetson path

Built from `laya-multilingual.s512.m16.fp32.ng.onnx`, md5
`191668c270e31fe4b94674b487bef1dc`, by a **manual `trtexec` command** — there was no build
script. The command below is reconstructed from the build log
(`logs/build_s512_fp16_nx.log`, lines 1 and 197):

```text
/usr/src/tensorrt/bin/trtexec --onnx=onnx/laya-multilingual.s512.m16.fp32.ng.onnx --fp16 \
    --saveEngine=out/laya.s512.fp16.engine --timingCacheFile=out/timing_s512.cache \
    --memPoolSize=workspace:2048
```

The s90 / s256 / s2048 / s8192 engines were built the same way — logs only, no scripts.
[`scripts/build_fp16_engine.sh`](scripts/build_fp16_engine.sh) wraps this command.

---

## 3. Reference and calibration data (committed here)

| Directory | Contents | Set md5 |
|---|---|---|
| `golden/` | ORT fp32 reference dump for the s512 bucket, 16 samples (8,040 B) | `9b1a961508ac24d249ee6d8dc143d0a6` |
| `calib/` | 8 calibration + 16 held-out samples at seq=512, plus `calib.txt` and `manifest.json` | `35db9f318c80a2ee7765b6091705b70c` |

Set md5 method: `find . -type f | sort | xargs md5 -q | md5 -q`.

> **Which reference pairs with which artifact.** The RK3588 deliverable is scored against the
> **NaN-guard-repaired** dump (`golden/ortdump.s512.json`, produced from
> `oldship.s512.m16.nanfix.onnx`). The nanguard dump from the same directory
> (`ortdump.oldship_s512.json`) outputs **NaN at every live marker position** in ORT fp32
> (80/256 values) and cannot be ranked at all. Do not use it.

---

## 4. Comparison and negative-result artifacts (not committed)

Kept for reference; each is cited by [`docs/quantization.md`](docs/quantization.md).

| File | Bytes | md5 | Note |
|---|---|---|---|
| `laya.s512.int8.engine` | 650,782,916 | `9c0ed68614a2832f8473c33338492c65` | implicit INT8 PTQ — **0/229 layers actually int8**, bit-identical to fp16 |
| `laya.s512.qdq.engine` | 525,664,180 | `8e54f4e04b179b310540152dc122eddc` | explicit Q/DQ, round 1 — top1 6/16 |
| `laya.s512.qdq.rs.engine` | 530,687,420 | `4666912542c1fcc38acff13e3db4cf1c` | explicit Q/DQ, round 2 — top1 5/16 |
| `laya.s90.fp16.engine` | 648,134,580 | `a7148fe1a6a756930d89c1c43723cd5a` | |
| `laya.s2048.fp16.engine` | 658,486,620 | `d4a04582459776a18249d9d106440c0a` | |
| `laya.s8192.fp16.engine` | 784,438,924 | `af911eccb11fb97db30e47d66e42279a` | external-weight ONNX build |
| `laya.s90.w4a16.engine` | 463,894,988 | `b1eeb76ef8629086a5af9faa469aefb8` | INT4 WoQ — **1.62× slower** than fp16 |

---

## 5. Rebuild environment

Pinned dependency manifests for both conversion environments, and the upstream revisions this
was built against, are in [`MANIFEST.md`](MANIFEST.md).
