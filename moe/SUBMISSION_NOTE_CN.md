# Submission Note

本目录已经整理成可提交状态。

## 提交主文件

优先提交根目录文件：

- `packed_solution.json`

根据 starter kit 的 FAQ：

- 如果 tag 根目录存在 `packed_solution.json`
- 评测流水线会直接使用它
- 不再依赖本地 `pack_solution.py`、`run_local.py` 等脚本

## 当前提交对象

- solution name: `moe_codex_20260425_m26a_safe_weakref_tgt1_cache`
- definition: `moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048`
- author: `ako4all-v2-codex`
- language: `python`
- entry point: `main.py::run`

## 源码布局

源码已按 starter kit 的 Python 约定放到：

- `solution/python/main.py`

配置文件已更新：

- `config.toml`

打包脚本已更新为支持：

- `language = "python"`
- `source_dir`
- `dependencies`
- `target_hardware`

## 本地测试说明

当前机器上的本地兼容测试方式见：

- `LOCAL_TESTING_CN.md`

注意：

- 本地 `RTX 4090` 和官方 `B200` 评测环境不同
- 本地为了保持 `flashinfer-bench run` 接口不变，启用了兼容环境变量
- 这些兼容改动是为了本地可测，不影响根目录 `packed_solution.json` 作为提交物被直接使用

## 提交前建议

1. 确认 `author` 是否要改成你们最终队名
2. 确认 git tag 打在包含本文件和 `packed_solution.json` 的提交上
3. 最终以 `packed_solution.json` 为准
