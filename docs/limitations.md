# 未验证 / 边界

本篇列出本次评测没有验证的方向与已知边界，避免把结论外推到未测范围。

## 未验证 / 边界

- **离群通道抑制类量化口径未试** —— 显式 Q/DQ 已证明 int8 能落核（见 [quantization.md](quantization.md)），
  失败在 per-tensor 激活 scale。per-channel 激活 / SmoothQuant 一类换口径的方向未验证。
- **Jetson s512 的 17.5 ms 大头未定位** —— attention 二次项只占 22%，长序列（2048，二次项占 66%）
  才是它的主战场。
- **RK3588 上 seq > 512 未测**。
- **`/dev/shm` 会被静默清空**（radxa，累计实测多次）——传输+测量须在一条 detached 驱动里原子完成。
- RK3588 交付依赖固定版本工具链（torch 2.14 + transformers 5.17 的导出链路，
  环境在 wsl2-local `/mnt/f/laya-oldship/.venv`）。**换导出环境必须重跑 16 条排序验收。**
