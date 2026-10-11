# Kev 实机 UI 验证证据（2026-10-11）

本组素材记录 Radxa 实际 Kev 服务上的最小 UI 验证。页面地址为 `http://100.77.150.16:18769/`，中文和英文页面均通过真实 `/api/health`、`/api/models` 与 `/api/decide` 请求验证。服务响应身份为 `model_id=kev`、`backend=rkllm`、`max_sequence_tokens=8192`、`max_options=16`。

## 素材

| 文件 | SHA-256 | 说明 |
|---|---|---|
| `kev-final-zh-result-20261011.png` | `1a78b98d392bf2b1c175f0fcf3fc75b44e8021f3cc95457be7cc1055af141e89` | 中文完成态，显示 Kev / rkllm / 8192 |
| `kev-final-en-pending-20261011.png` | `abe18a89512700cf46c742d6a495643f70a59661badbde6b360c052082654741` | 英文真实请求进行中，显示 Running 与 spinner |
| `kev-final-en-result-20261011.png` | `f574485d31c02750904aab42807f38a382337921d3505e22dcaf4efa740911cb` | 英文完成态，显示 Kev / rkllm / 8192 |
| `kev-final-qa-20261011.gif` | `308b3d77b5030b83b95d4277d9b7183c5b30fb69c004f8c4409b8809b1803748` | 中文完成、英文 pending、英文完成三帧 montage |
| `kev-final-evidence-20261011.json` | `e6d496da79d2d4119849b202e9b9faa6a19392e9ad86137a0dde570b4122cda6` | URL、时间、健康状态、请求阶段和素材 SHA |
| `kev-final-zh-result-20261011.yml` | `c45e65c63fa86f0ee9fcbabd1d2375f9ef6022aa60e9e3f59b45f2d6ca474d3f` | 中文 AX 快照，保留 API 原始分数与毫秒值 |
| `kev-final-en-result-20261011.yml` | `09312859910f5f975838777213acb2e9c02b4f93a3391eeb1c1a2aceca1bbbac` | 英文 AX 快照，保留 API 原始分数与毫秒值 |

截图和 GIF 属于 **pre-rounding** 页面素材：它们在 UI 显示格式化提交前录制，截图中的长小数与 API 原始值一致。后续 UI 提交只把展示分数格式化为 3 位小数、耗时格式化为 1 位小数，API 与证据 JSON 保留原始值。

## 真实操作记录

- 中文：`降低三号输送带速度`，选择 `降低输送带速度`，原始 score `0.9032248287821506`，total `721.7322869691998 ms`。
- 英文：`Reduce conveyor 3 speed to 60%`，选择 `Reduce conveyor speed`，原始 score `0.9236911123535364`，total `481.79490701295435 ms`。
- pending 帧是在浏览器捕获到同一次 `/api/decide` POST 后立即保存；它用于证明真实请求期间 UI 保持 pending 状态。
- Playwright 网络记录确认三次请求均返回 HTTP 200：`GET /api/health`、`GET /api/models`、`POST /api/decide`。工具仅提供方法、路径和状态码，未把未读取的 response body 写入证据。

## 复核命令

```bash
sha256sum docs/kev-final-zh-result-20261011.png \
  docs/kev-final-en-pending-20261011.png \
  docs/kev-final-en-result-20261011.png \
  docs/kev-final-qa-20261011.gif \
  docs/kev-final-evidence-20261011.json
curl -fsS http://100.77.150.16:18769/api/health
curl -fsS http://100.77.150.16:18769/api/models
```

浏览器素材的 `app.js` served SHA 为 `870dfe9abcac8ae6556f2b35e61987857babd6adc9a27b06c2e5ec745640b358`；截图录制时使用的是同一 served web source。该证据覆盖 Kev 实机的中文完成态、英文 pending 态和英文完成态；Laya 实际切换待其服务可用后另行记录。
