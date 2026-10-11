# Kev 500-token 上下文基准

本基准测量 Kev RKLLM 命令选择路径在固定 500 个 assembled tokens 上下文中的行为。每条冻结语义请求后追加固定的、中性的合成工业文档元信息句子。追加内容不描述动作、设备状态、传感器状态或运行条件；原始 input、prompt 语义、候选项、期望标签、语言和类别保持不变。该追加只用于控制长度，不代表自然长任务的准确率结论。

数据集包含 120 条冻结语义 case：英文 60 条、中文 60 条，覆盖 direct、paraphrase、negation、ambiguous、out-of-scope。每条 case 按原候选顺序和反转候选顺序各发送一次，每台设备共 240 条计分请求。计分前发送 5 条 warmup，warmup 不计入 240 的分母；warmup 失败时立即停止本轮。

## 可复现输入

- [`context500-cases.jsonl`](context500-cases.jsonl) 保存请求体、期望标签、两种顺序、精确 token 数和逐请求 SHA256。期望字段只存在于元数据，不会发送到 HTTP body。
- [`context500-metadata.json`](context500-metadata.json) 保存冻结源数据 SHA、部署 tokenizer SHA、Kev codec 源 SHA、delimiter ID、目标长度以及 generator/runner SHA。
- [`tokenizer.fe000.json`](tokenizer.fe000.json) 是从 Radxa 发布部署复制并校验的 tokenizer，SHA256 为 `fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927`。
- [`generate_context500.py`](generate_context500.py) 对每条生成请求调用应用内 `KevCodec.encode`，并与独立 token 计数交叉校验；不一致时拒绝生成。
- [`run_context500.py`](run_context500.py) 通过 loopback 顺序发送 HTTP 请求，恢复已有结果前校验 dataset/request SHA，不自动重试，每条计分请求写一条原始 JSONL 并生成 summary。

每台设备的结果必须记录实际 image、model、tokenizer、runtime 和 Hugging Face revision。发布 pin 与 artifact 证据位于对应 release evidence 目录；本基准不下载或替换部署 artifact。

## 测量范围

client elapsed 是 loopback HTTP 请求外层的计时；服务端 `encoding_ms`、`inference_ms`、`total_ms` 单独统计。summary 对 client 端到端时间及每个可用的服务端时间字段分别报告 n、mean、minimum、p50、p95、maximum。HTTP 错误、超时和缺少任一服务端时间字段的响应单独列出；缺失时间不会按零计入。

准确率使用同一批 240 条计分请求，分别报告总体、语言、类别和候选顺序。summary 包含完整错误清单，以及 HTTP 200 但选中项不同于期望标签的 `wrong_predictions`。5 条 warmup 不进入准确率或性能分母。服务端每次请求清理 KV cache，不共享请求历史；runner 串行发送请求。

硬件结果只会在设备 worker 生成逐请求原始 JSONL、程序 summary 和 artifact/hash 证据后写入 `rk3576/` 与 `rk3588/`。这些文件出现前，本文件描述的是测试方法和范围，不代表设备结果已经完成。
