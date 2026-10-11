# RK3588 发布验证（2026-10-11）

本目录记录 source `ef82f8360c2b787f97c958f89c92fe569c62ecb1` 在 ready inventory、HF revision `db0545065d4da3598d009407fc3bd0c7632ca199` 上的 stock 部署验证。

- 服务地址：`http://<device-ip>:18768`
- 镜像标签：`edge-command-workbench:rockchip-rkllm-0.3.1-review`
- archive SHA-256：`251cb7fef080f489bf311b37c5bddbba1cd72e887e71f10ce12fc25a1fe64f77`
- image config SHA-256：`d7a81c4f8ef3eaa2c21560b2d37939c6088357ec0c1d41c3089ff5fbc3836de9`
- bootstrap 退出码：`0`
- 产物统计：13 个目标文件从已校验 SHA 的缓存复用，另从 `https://hf-mirror.com` 下载 1 个新 image archive；这不是全新下载 14 个文件。
- health 返回 `model_ready=true`、`backend=rkllm`、`model_id=kev`、`max_sequence_tokens=8192`、`max_options=16`。
- 相同的短决策请求对 `model_id=kev`（`rkllm`、8192）和 `model_id=laya`（`rknn`、90）均返回 HTTP 200；Laya 使用已有远程 profile `http://127.0.0.1:18771`。
- image 内置 helper `/opt/kev/libkev_runtime.so` 已校验，SHA-256 为 `022e0102f9dd4c8839ba7acbc68f03d8811bf020cde84d6cf1343829cd66f919`；没有使用外部 helper mount。

此前的 `a556`/`d870` 14 文件 bootstrap 记录保留为历史证据。此次验证不包含 voice 和 ASR。证据目录只包含选定元数据及原始请求、响应、耗时文件，不包含模型权重和 image archive。

## 当前支持范围

当前支持范围保留 RK3576 RKNN 和既有 RK3576 Kev 2K 路径；8K cat-remote 运行单独记录，超限输入在实际 2088 tokens 后由 guard 拒绝。RK3588 Kev 8K 已由本目录的 stock release evidence 验证。Jetson 仍是历史 native Kev 路径，本工作台不将其声明为已就绪部署目标。
