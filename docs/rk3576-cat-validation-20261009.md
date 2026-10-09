# RK3576 cat-remote 真机验证记录（2026-10-09）

本记录对应 `cat-remote` 上的 LubanCat3 RK3576 真机验证。记录只覆盖本次实际执行的模型加载、RKNN 推理、服务 API、heldout 排序对比和资源观察；语音链路、设备真实命令执行及现场语义准确率不在本次证据范围内。

## 环境与产物

- 设备：`cat-remote`，LubanCat3，RK3576
- RKNN SDK：`2.3.2`
- RKNN driver：`0.9.8`
- 服务地址：`127.0.0.1:18768`
- 新模型：`/home/cat/laya-workbench-validation-20261009-a1/model/laya-multilingual.s90.m16.fp16.nanfix.rk3576.rknn`
- 新模型 SHA256：`fe2300f034b11d98a69f686ea35b7889db813f5cf227bac4afcd27a0e77c7405`
- 新模型大小：`658328875` bytes（RKNN dump 中报告 `627.83 MB`）
- 对照旧模型 SHA256：`6b020fd37bc7190564ef88ffe8880f659e8800afafa9ff2333beedfa9dc451db`
- 新模型使用静态 shape：`input_ids [1,90]`、`attention_mask [1,90]`、`marker_pos [1,16]`、`marker_mask [1,16]`、`qtype [1]`；输出为 `logits [1,16]` 和 `act_logits [1,2]`。

新服务启动、health 和请求记录见 `new_app_18768.log`。启动时出现的 `RKNN_QUERY_INPUT_DYNAMIC_RANGE` warning 与静态 shape 模型相符，随后服务启动完成并返回 health `200`；日志中服务请求返回 `200` 或按输入校验返回 `422`，没有记录启动失败。

## 真机推理与延迟

### Native benchmark

`native_bench.json` 使用 warmup `10`、计时迭代 `120`。runner 的 `README.txt` 记录了实现细节：benchmark 在 warmup/iteration 循环前固定取 `result['requests'][0]`，因此 120 条 raw 记录的 `input_token_length` 全部为 `43`，且全部重复第一条 `Reduce conveyor 3 speed to 60%.` 请求。这里只能说明同一条 43-token 输入的重复推理延迟，不能把它表述为三条请求的轮询 benchmark，也不能从中推出三条不同语义输入的延迟分布。

| 指标 | p50 | p95 | min | max |
| --- | ---: | ---: | ---: | ---: |
| encoding | 0.4338 ms | 0.4849 ms | 0.4246 ms | 0.4976 ms |
| inference | 334.9951 ms | 340.5998 ms | 327.5242 ms | 344.4669 ms |
| total | 335.4297 ms | 341.0384 ms | 327.9613 ms | 344.9611 ms |

这是新模型在真实 RK3576 上的 120 次重复 benchmark。旧证据 `old_artifact_bench.json` 是 synthetic 输入（全零 `input_ids`、固定 mask/marker），其 p50 为 `337.5725597143173 ms`；两者输入和实验类型不同，应分开记录，不能直接当作同一 benchmark 的 before/after。

### Heldout 真机 dump

`live_rk3576.json` 对 16 个 heldout 样本各执行 10 次 warmup 后测 1 次，RKNN runtime 报告 SDK `2.3.2`、driver `0.9.8`、target `rk3576`，模型为 static shape，`core_mask=auto`。延迟为 `n=16`、p50 `350.6 ms`、mean `350.97 ms`、min `344.79 ms`、max `362.23 ms`，无 errors、无 warnings（除静态 shape 查询 warning，见上）。

## 排序对比

`rank_comparison.json` 将 16 个 heldout 样本的真机 RK3576 logits 与 `reference=/home/cat/laya-workbench-validation-20261009-a1/golden/oldlin_ref_normalized.json` 对比：

- top-1/reference 首选一致：`16/16`
- 全部候选排序完全一致：`15/16`
- 不完全一致的是 sample `03`；两边首选均为 option `1`，差异出现在较低排名候选的相对顺序
- Kendall tau mean `0.998106`，min `0.969697`
- relative L2 mean `0.441127`，max `1.24129`
- max absolute difference `0.732651`
- `act_logits` relative L2 max `0.048092`
- 工具给出的整体 verdict 为 `PARTIAL`

这里的 `16/16` 是“与参考输出的首选一致性”，不是语义准确率；`15/16` 是与参考的全排序一致性。本次只对比参考输出，不作语义准确率评价。也不据此宣称所有模型算子都运行在 NPU 上；本次检查的 sysfs 路径未暴露可用的 NPU utilization counter。

## 服务 API 与语义样例

新服务的三个请求均通过 `POST /api/decide` 返回 HTTP `200`，返回 `request_id`、有限的 scores、timings、`model_ready=true`、`backend=rknn`、`max_sequence_tokens=90`、`max_options=16`。本次 API 样例的实际选择如下：

| 输入 | 实际首选 | total |
| --- | --- | ---: |
| `Reduce conveyor 3 speed to 60%.` | `s90` / Set speed | 339.1457 ms |
| `Show camera 2 temperature.` | `stop` / Stop | 330.4736 ms |
| `停止三号输送带。` | `stop` / 停止 | 322.4031 ms |

第二条英文输入的实际首选是 `stop`，不是 `status`；这是一个应保留在验证记录中的错误语义样例。它不能用于证明 camera temperature 识别正确。

后续 `current_codec_ort_three.json` 对同三条请求的 codec+FP32 对照显示：speed option id 为 `speed`（语义为 Set speed），camera temperature 仍选择 `stop`，中文停止仍选择 `stop`；其中 camera FP32 active logits 为 `[-3.4307, -2.5169, -3.6338]`。当前 codec+FP32 也复现 camera 误选，证据不支持将其单独归因于 RKNN 量化；原 codec parity 尚未完成。

显式状态样例在 `explicit_api.json` 中分别用中英文候选表达“status”：

- `Read current conveyor status.` → `status`，total `327.7537 ms`
- `查询三号输送带运行状态。` → `status`，total `325.4407 ms`

这两条只演示在明确的中英文“状态”候选集合下模型选择 `status`，不等于现场设备状态识别准确率。此次没有执行语音输入，也没有执行任何真实设备命令；`stop`、`s90`、`status` 都只是 API 候选 option id。

服务输入校验也有实证：2000 字符输入返回 HTTP `422`，错误码 `input_too_long`，assembled input `161` tokens、上限 `90`；重复 option id 返回 HTTP `422`，错误信息为 `option ids must be unique`。

## 运行时资源与服务状态

`new_activity_sampling.txt` 记录了 5 次请求前后采样：NPU frequency 始终为 `950000000`，CPU 使用率从 `13.3%` 增至 `14.6%`，温度从 `49922 mC` 到 `51769 mC`，memavail 约 `1197656` 到 `1197484 kB`。NPU frequency 是频率，不是 utilization；本次检查的 sysfs 路径未暴露可用的 NPU 利用率计数器，不能以这些数据宣称 NPU 利用率或算子放置。

`/tmp/cat-device-validation/new/final_state.txt` 的只读采样时间为 `2026-10-09T16:31:50+08:00`：`18768` 已无 listener，说明 owned API 已停止；根文件系统 `/` 可用磁盘空间为 `1.4G`（总计 `58G`，已用 `54G`，98%）；`MemAvailable` 为 `2852696 kB`。当时既有的 `conversational-voice-speech` 和 `xiaozhi-display` 两个 container 均为 `healthy`。此前采样期间 memavail 约 `1.2 GB` 的记录仍见 `new_activity_sampling.txt`；这些是本次会话的设备状态观察，不是长期资源容量结论。

## 文件传输与版本对照

原计划的 Native transfer 中转已停止。最后使用 Python fallback 进行真实直传，日志记录 `python direct transfer exit=0`，并在目标端校验 `md5: d87a4c7e320dd53d98b9965bded404d7`，见 `/tmp/cat-transfer-diagnosis/python-direct-transfer.log`。模型文件的最终 SHA256 以本记录“环境与产物”中的值为准。

旧证据来自 `/tmp/cat-device-validation/pulled-old/`：旧模型可 load/init，使用同一 SDK/driver，旧 heldout 对照也记录过 `16/16` top-1 和 `15/16` exact order；本记录的新结果使用新模型和新的真机 dump，不能把旧 synthetic benchmark、旧 artifact dump 与新真实重复 benchmark 混为一组。

## 复现实验配置

真机 heldout dump 使用：

```text
/home/cat/rkvoice-stream/.venv/bin/python
/home/cat/laya-workbench-validation-20261009-a1/app/validation/runner/47_device_dump.py
  --rknn /home/cat/laya-workbench-validation-20261009-a1/model/laya-multilingual.s90.m16.fp16.nanfix.rk3576.rknn
  --name rk3576_live
  --heldout /home/cat/laya-workbench-validation-20261009-a1/heldout
  --out /home/cat/laya-workbench-validation-20261009-a1/results/new/live_rk3576.json
  --core-mask auto --warmup 10 --iters 1
```

排序对比使用：

```text
/home/cat/rkvoice-stream/.venv/bin/python
/home/cat/laya-workbench-validation-20261009-a1/app/validation/runner/46_rank_from_dumps.py
  --ref reference=/home/cat/laya-workbench-validation-20261009-a1/golden/oldlin_ref_normalized.json
  --dump rk3576_live=/home/cat/laya-workbench-validation-20261009-a1/results/new/live_rk3576.json
  --heldout /home/cat/laya-workbench-validation-20261009-a1/heldout
  --out /home/cat/laya-workbench-validation-20261009-a1/results/new/rank_comparison.json
```

native benchmark 配置为模型路径 `/home/cat/laya-workbench-validation-20261009-a1/model/laya-multilingual.s90.m16.fp16.nanfix.rk3576.rknn`、warmup `10`、iterations `120`；完整 raw 结果见 `/tmp/cat-device-validation/new/native_bench.json`。服务/API、显式状态、运行时采样和服务日志分别见：

```text
/tmp/cat-device-validation/new/service_api.json
/tmp/cat-device-validation/new/explicit_api.json
/tmp/cat-device-validation/new/new_activity_sampling.txt
/tmp/cat-device-validation/new/new_app_18768.log
/tmp/cat-device-validation/new/live_rk3576.json
/tmp/cat-device-validation/new/rank_comparison.json
```

## 尚待补充

same3inputs 的 codec+FP32 三请求对照已有结果，见 `/tmp/cat-device-validation/wsl-parity/current_codec_ort_three.json`；它复现了 speed、camera、中文停止三条请求的选择，其中 camera 仍为 `stop`。`/tmp/cat-device-validation/wsl-parity/tokenizer_probe.stdout` 与 `tokenizer_probe.stderr` 只记录到 `before_import`，原 codec parity 尚未完成，不能把当前结果写成 codec 完全一致。后续若补齐 parity，应更新同一文件中的对应结论，并保持新真实 benchmark、旧 synthetic benchmark、heldout 参考排序和 API 语义样例分开。

## 产品侧观察

产品侧另有 codec `encode` 抛出 `NativeError` 后由 `decision.py` 修复为 HTTP `422` 的验证记录；该问题与模型文件传输无关。本次传输记录仍是：Native transfer 中转已停止，Python fallback 真实直传成功并完成 checksum 校验，见 `/tmp/cat-transfer-diagnosis/python-direct-transfer.log`。
