# RK3588 Edge Command Router 部署证据

本报告记录候选提交 `d5527aa5b155c23c9f03c61d4dc0ccc8936261ae` 通过 `sensecraft-engine` 0.0.3 部署到 Radxa RK3588 设备的脱敏证据。Spark daemon 也报告 engine 0.0.3。部署 ID 为 `1b90b989-5ea0-4702-8b1d-c30673585060`，终态为成功，退出状态为 0，耗时约 59.3 秒。

成功轮复用了已准备的 archive cache，并记录 13 个 model/shared 文件的 checksum 命中。engine 事件同时验证了精确 image tag 与 config digest，并记录跳过 Docker load。本报告不把该轮描述为全新模型下载或 cold-start 构建。候选镜像达到 `model_ready=true`、`backend=rkllm`、`model_id=kev`、`max_sequence_tokens=8192`、`max_options=16`。已有 Laya profile 继续可用；它是兼容 profile，不构成新的 RK3588 准确率结论。

## 已覆盖的验证

- 5 个固定 smoke request 均返回 HTTP 200 且选择结果符合预期：`jev-001-en`/`jev-001-zh` → `set_speed`，`jev-002-en`/`jev-002-zh` → `status`，`jev-003-en` → `start`。
- 40025-token fixture 在进入 NPU 前被部署容器拒绝，返回 HTTP 422、`input_too_long`、`actual_tokens=40025`、`max_tokens=8192`。
- 这 5 个 smoke request 用于部署检查，不是模型准确率基准；未测试 ASR 和语音行为。

## 失败与回滚记录

第一次尝试（`b49ae360`）进入 bootstrap 后因无法写入 `.cache/rkllm-artifact-plan.tsv`（`Permission denied`）失败；没有证据把 offline package gate 归属于该 deployment。第二次尝试（`a93c7ccd`）验证了归档和 13 个 model/shared 文件的 checksum 并跳过 Docker load，随后 Docker mirror rewrite 将本地镜像引用改写到不存在的 `docker.m.daocloud.io/library/...` 路径，服务启动阶段失败。两次尝试之间的权限修复无法由现有事件确认。回滚后观察到旧服务健康；本文不推断具体回滚命令。成功候选轮使用修正后的本地归档处理，之后恢复 daemon；未继续重试。

成功部署请求没有携带 `network_region` 或 `force_offline` override，这些值来自候选设备 YAML。部署使用 RK3588 remote preset，并保留已有 Laya service profile。本报告只覆盖 RK3588，不代表 Jetson native 或 RK3576 硬件结果。

## 脱敏证据

- [成功部署事件](evidence/deploy-1b90b989-events.compact.ndjson)
- [成功轮 bootstrap 事件](evidence/bootstrap-1b90-engine-derived.ndjson)
- [候选请求](evidence/deploy-candidate-request.sanitized.json)
- [Health 结果](evidence/health.final.json) 与 [模型列表](evidence/models.final.json)
- [5 个 smoke 请求](evidence/goldens5.requests.sanitized.ndjson) 与 [响应](evidence/goldens5.responses.ndjson)
- [超长输入 guard 证据](evidence/overlength8192.evidence.txt)
- [重新审计的失败时间线](evidence/reaudit-timeline.sanitized.md)、[b49 bootstrap 事件](evidence/b49ae360.reaudit.sanitized.ndjson)和 [a93 bootstrap/镜像事件](evidence/a93c7ccd.reaudit.sanitized.ndjson)
- [脱敏部署报告](evidence/deployment-1b90b989-report.sanitized.md)

归档未包含主机名、私有路径、凭据和 raw/private 抓取文件。
