# Edge Workbench 多平台 Native 验证 / Multiplatform Native Validation

验证日期：2026-10-09。源码和构建环境在 Spark；推理在目标设备本地执行。本文记录 native backend gate，不能替代 Docker 镜像部署、设备控制链路或工业现场闭环验收。

Validation date: 2026-10-09. Spark is the source and build workstation; inference runs locally on each target device. This document records the native backend gate only. It is not a Docker deployment, device-control, or industrial end-to-end acceptance report.

## Source and artifacts / 源码与产物

- Spark app source snapshot used by the historical probes: `app/edge_workbench/native.py`, SHA256 `6cd1d9c67c63caef3a8fc2789e547e65a917e9a9d9957f6a8b597bbd4187af30`.
- The fresh reproducible RK3588 CLI sample uses the current worktree `native.py`, SHA256 `5c434fa631efe7b60ea32353707ca2ee57c1adcba2e5772042928a192c2fa152`; this hash is recorded in `rk3588-cli-validation.json`.
- Shared tokenizer: `tokenizer.json`, SHA256 `609d8f4c067cd3950f88594c5a802616cea245823836ef5848ee4fc40aab5b6f`; config SHA256 `6c6b2d8e3c84ce0e671c129cd6b374b235d6f9863042a5836358d00a89bbb5a1`.
- RK3588 archive: `tmp/edge-workbench-0.2.0-candidate/models/rk3588/model.rknn`, SHA256 `a194c9504d38173525fb1aedb513c3261c01ff4860c9d4a0743abdc953b88b97`.
- Jetson archive: `tmp/edge-workbench-0.2.0-candidate/models/jetson/model.engine`, SHA256 `e95ca6a6c5824c54c7e7c6959a7ccff98d0d85139b915751dfec23ee132f5068`.
- RK3576 uses the already-published 0.1.0 HF artifacts; this validation did not download or duplicate them.

## Results / 结果

| Target | Runtime | Device profile | Native result | Performance | Scope |
|---|---|---|---|---:|---|
| RK3576 cat | RKNN Lite 2.3.2 / driver 0.9.8 | RK3576 | Prior hardware rank comparison: heldout top-1/reference 16/16; full candidate order 15/16; separately recorded API examples (3 initial + 2 explicit status), with no aggregate semantic score | Performance not re-measured in this archive | Hardware rank comparison is partial; 16/16 simulation is offline only; Docker not re-run here |
| RK3588 radxa | RKNN Lite 2.3.2 / driver 0.9.8 | Rock 5T, `nvidia,p3767` not applicable | 6 semantic cases: 5/6; all active logits finite | **p50 278.52 ms, p95 280.06 ms** | Fresh CLI validation; one low-score semantic counterexample, no runtime load failure |
| Orin NX | TensorRT 10.3.0 | `nvidia,p3767-0000` | 6 semantic cases: 5/6; all active logits finite | **p50 8.04 ms, p95 12.54 ms** | Reproducible CLI native gate; Docker not loaded |
| Orin Nano | TensorRT import only | `nvidia,p3767-0003` | TensorRT/numpy/tokenizers/CUDA imports passed | Not measured | Engine inference intentionally skipped: initial available memory 1.8 GiB, below the 2.5 GiB safety gate |

当前 RK3588 CLI 采样把 prompt、options、输入、模型 hash、源码 hash 和结果写入 JSON；Orin NX CLI JSON 记录 prompt、options、输入、模型 hash 和结果，但不包含 native.py 源码 hash。旧 RK3588 native probe 的 p50 279.42/p95 282.37 ms，以及旧 Orin NX native probe 的 p50 8.45/p95 12.97 ms，保留在 raw evidence 中，属于历史独立采样，使用历史 native.py SHA `6cd1d9c...`，不与当前 CLI 输入强行配对。设备验证证据中的历史 engine/tokenizer SHA 也不代表当前 Docker candidate；当前 build freeze 使用 native.py SHA `5c434fa631efe7b60ea32353707ca2ee57c1adcba2e5772042928a192c2fa152`，candidate image/archive 的 digest 与 SHA 另见构建证据。

The current RK3588 CLI sample records prompt, options, inputs, model hash, source hash, and results in its JSON; the Orin NX CLI JSON records prompt, options, inputs, model hash, and results, without a native.py source hash. The earlier RK3588 native-probe values (p50 279.42/p95 282.37 ms) and earlier Orin NX native-probe values (p50 8.45/p95 12.97 ms) remain in raw evidence as independent historical samples using native.py SHA `6cd1d9c...`; they are not paired with the current CLI inputs. Historical engine/tokenizer hashes in the device evidence do not identify the current Docker candidate. The current build freeze uses native.py SHA `5c434fa631efe7b60ea32353707ca2ee57c1adcba2e5772042928a192c2fa152`; candidate image/archive digests and hashes are recorded separately in the build evidence.

The recurring semantic counterexample is `Show camera 2 temperature`, which selects `pause` under the current option set. This is recorded as model behavior for follow-up evaluation; it is not counted as a runtime failure.

重复出现的语义反例是 `Show camera 2 temperature`，当前选项集合下被选择为 `pause`。这里记录的是模型行为，后续需要补充数据或调整选项定义；不把它归为 runtime 加载失败。

## Measurement contract / 测量约定

The semantic prompt and options are fixed in `scripts/validate_workbench_native.py`. The six cases are three English and three Chinese commands. The performance loop runs five warmup calls followed by twenty calls to the same `检查设备状态` request. It measures native inference through the current Spark `native.py` adapter, including its input preparation and output extraction, not HTTP or UI latency and not an industrial command execution.

语义 prompt、候选项和六条测试输入固定在 `scripts/validate_workbench_native.py`。性能测试先 warmup 5 次，再对同一条 `检查设备状态` 请求执行 20 次。性能只测当前 Spark `native.py` adapter 的 native inference（含输入准备和输出提取），不包含 HTTP/UI 延迟，也不代表工业命令执行时延。

For p50/p95, the probe sorts the twenty measured latencies and selects `sorted[n//2]` for p50 and `sorted[min(n-1, int(n*0.95))]` for p95. With `n=20`, the p95 index is 19, so the reported p95 equals the maximum sample. This is an explicit probe convention rather than a `numpy.percentile` interpolation result.

p50/p95 的计算是对 20 个延迟排序后，p50 取 `sorted[n//2]`，p95 取 `sorted[min(n-1, int(n*0.95))]`。当 `n=20` 时 p95 索引为 19，因此报告的 p95 等于最大样本。这是 probe 的明确约定，不是 `numpy.percentile` 插值结果。

## Reproduction / 复现

The probe requires an existing vendor runtime and a local model/tokenizer. It does not install packages, download weights, start or stop services, load Docker images, or issue device commands. Example:

该 probe 要求目标设备已经提供 vendor runtime、本地模型和 tokenizer。脚本不会安装依赖、下载权重、启停服务、加载 Docker 镜像或执行设备控制命令。示例：

```bash
python3 scripts/validate_workbench_native.py \
  --backend tensorrt \
  --model /home/harvest/laya-trt/out/laya.s90.fp16.engine \
  --tokenizer-dir /dev/shm/jetson-workbench-native-20261009/tokenizer \
  --native /dev/shm/jetson-workbench-native-20261009/code/native.py \
  --warmup 5 --iters 20 \
  --output /dev/shm/jetson-workbench-native-20261009/validation.json
```

For an offline contract check, use `--dry-run`; `--seq-len 0` is rejected. The script reports the model hash, prompt, options, input shapes, semantic rows, and the p95 method in JSON.

离线检查可以使用 `--dry-run`；`--seq-len 0` 会被拒绝。脚本 JSON 会记录模型 hash、prompt、选项、输入 shape、语义结果和 p95 计算方式。

## Evidence / 证据

Small raw logs and JSON are stored under `docs/evidence/edge-workbench-20261009/`. The archived model files are under `tmp/edge-workbench-0.2.0-candidate/models/`. The evidence contains no credentials. The RK3576, RK3588, Orin NX, and Orin Nano records are separate; the Nano record is import-only by design.

小体积 raw log 和 JSON 位于 `docs/evidence/edge-workbench-20261009/`，模型归档位于 `tmp/edge-workbench-0.2.0-candidate/models/`。证据不包含凭据。RK3576、RK3588、Orin NX、Orin Nano 的记录彼此独立；Nano 记录按设计只覆盖 import。
