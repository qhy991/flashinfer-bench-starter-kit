# Local Testing

这份说明用于在当前这台本地机器上，继续使用官方 `flashinfer-bench run` 接口做 MoE 测试。

## 背景

本地 `RTX 4090` 环境和官方 B200 评测环境有两个差异：

- 官方 MoE definition 自带的 Python reference 在 4090 上容易 OOM
- 当前容器权限会让 `flashinfer-bench` 的多进程 CUDA IPC 路径触发 `pidfd_getfd: Operation not permitted`

为保持命令接口不变，本地兼容方案是：

- 仍然使用官方 `flashinfer-bench run ...`
- 通过环境变量启用单进程 runner
- 通过环境变量把本地 reference 替换为 4090-safe 的 streaming baseline

## 1. 生成本地 trace root

```bash
/root/mlsys_flashinfer_challenge/.venv/bin/python \
  ~/flashinfer-bench-starter-kit/scripts/prepare_local_trace_root.py \
  --dataset /root/mlsys_flashinfer_challenge/smoke_dataset \
  --solution-json /root/mlsys_flashinfer_challenge/smoke_dataset/solutions/torch_compile_moe/moe_codex_20260425_m26a_safe_weakref_tgt1_cache.json \
  --output ~/flashinfer-bench-starter-kit/.local_eval/moe_m26a_smoke
```

如果要测试 full dataset，把 `--dataset` 改成：

```bash
/root/mlsys_flashinfer_challenge/mlsys26-contest
```

## 2. 设置本地兼容环境变量

```bash
export FIB_FORCE_SINGLE_PROCESS=1
export FIB_OVERRIDE_REFERENCE_SOLUTION_JSON=/root/mlsys_flashinfer_challenge/smoke_dataset/solutions/torch_compile_moe/moe_torch_compile_streaming_baseline_v2.json
```

说明：

- `FIB_FORCE_SINGLE_PROCESS=1`
  - 保持 `flashinfer-bench run` 命令不变
  - 但内部改为单进程执行，避开本机的 CUDA IPC 权限限制

- `FIB_OVERRIDE_REFERENCE_SOLUTION_JSON=...`
  - 保持 `flashinfer-bench run` 命令不变
  - 但内部把 definition 里的 reference 替换成 4090-safe baseline 代码

## 3. 使用官方接口运行

Smoke：

```bash
/root/mlsys_flashinfer_challenge/.venv/bin/flashinfer-bench run \
  --local ~/flashinfer-bench-starter-kit/.local_eval/moe_m26a_smoke \
  --definitions moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048 \
  --solutions moe_codex_20260425_m26a_safe_weakref_tgt1_cache \
  --save-results --use-isolated-runner --log-level INFO --timeout 300 \
  --atol 1 --rtol 0.3 --required-matched-ratio 0.9
```

注意：

- 命令接口和官方一致
- `--use-isolated-runner` 可以保留
- 本地兼容模式由环境变量接管，不需要修改命令参数

## 4. 当前已验证结果

在当前机器上，上述 smoke 命令已跑通，trace 保存在：

`/root/mlsys_flashinfer_challenge/ako4all_v2_framework/reports/starterkit_m26a_smoke_eval_20260425/traces/moe/moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048.jsonl`

本地兼容模式下，该 workload 的结果为：

- status: `PASSED`
- speedup: `13.79x`

## 5. 适用范围

这套方式的目标是：

- 保持官方 CLI 接口不变
- 让本地 4090 环境可以正确执行 MoE 测试

它不是官方最终 B200 评测环境的完全复刻；它是“本地可跑、接口一致、判分路径尽量一致”的兼容测试方式。
