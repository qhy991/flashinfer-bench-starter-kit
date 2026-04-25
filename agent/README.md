# Full-Agent Runner

This directory contains a minimal Codex-driven optimization agent for the
MLSys 2026 FlashInfer AI Kernel Generation Contest full-agent track.

## What It Does

The runner:

1. Creates a writable working copy for a target kernel directory.
2. Runs the target's local benchmark command.
3. Starts a Codex non-interactive optimization session with task instructions.
4. Re-runs the benchmark.
5. Resumes the same Codex session with a "continue optimizing" style prompt.
6. Repeats until the iteration limit or a stop condition is reached.

This is intentionally simple. It does not hardcode final kernel solutions and
does not embed final answers into prompts.

## Layout

- `run_agent.py`: main orchestration entrypoint
- `configs/targets.json`: target-specific benchmark commands and metadata
- `prompts/system_prompt.md`: high-level operating rules for Codex
- `prompts/continue_prompt.md`: continuation prompt template
- `tools/run_benchmark.py`: benchmark wrapper used by the agent loop

## Example

```bash
cd /home/qinhaiyan/flashinfer-bench-full-agent-submission

python3 agent/run_agent.py \
  --target dsa_indexer \
  --max-iters 3 \
  --work-root /home/qinhaiyan/flashinfer-agent-runs
```

## Environment

Expected tools:

- `codex` CLI
- Python 3
- `FIB_DATASET_PATH` for benchmark execution

## Output

Each run creates:

- a work directory under `--work-root`
- benchmark artifacts under `agent_runs/<target>/benchmarks/`
- Codex transcripts under `agent_runs/<target>/codex/`
- a machine-readable state file `run_state.json`

## Notes

- The runner uses `codex exec` for the first turn and `codex exec resume --last`
  for follow-up turns.
- The work directory is isolated per run to keep the original checked-in target
  directories clean.
