# KernelMind Full-Agent Submission

This repository is prepared for the **MLSys 2026 FlashInfer AI Kernel Generation Contest**
as a **Full-Agent** submission.

It contains:

- final kernel directories in starter-kit-compatible top-level subfolders
- an `agent/` directory that drives Codex-based iterative optimization
- supporting starter-kit scripts and docs needed for local reproduction

## Submission Type

- Approach: `Full-Agent`
- Team / Author: `KernelMind`

If shared with organizers, this repository should be explicitly labeled as the
**Full-Agent** submission repo.

## Top-Level Kernel Directories

- `dsa_attention/`
- `dsa_indexer/`
- `gdn_decode/`
- `gdn_prefill/`
- `moe/`

Each subdirectory has its own:

- `config.toml`
- `solution/`
- `scripts/`

There is intentionally **no root-level `config.toml`** so the evaluation pipeline
does not treat the repository root as an extra solution.

## Current Validation Status

Verified in the current local environment:

- `dsa_attention`: full benchmark passed
- `dsa_indexer`: full benchmark passed
- `gdn_decode`: smoke benchmark passed
- `gdn_prefill`: smoke benchmark passed

## Full-Benchmark Results Collected So Far

- `dsa_attention`
  - full workloads passed: `23/23`
  - mean runtime: `0.1959 ms`
  - mean speedup: `42.51x`

- `dsa_indexer`
  - full workloads passed: `128/128`
  - mean runtime: `3.6459 ms`
  - mean speedup: `1.84x`

- `gdn_decode`
  - smoke workloads passed: `2/2`
  - mean runtime: `0.005376 ms`
  - mean speedup: `427.73x`

- `gdn_prefill`
  - smoke workloads passed: `2/2`
  - mean runtime: `0.235776 ms`
  - mean speedup: `85.83x`

## Agent

The `agent/` directory contains a minimal Codex-driven orchestration loop that:

1. creates a writable working copy for a target
2. runs benchmark commands
3. invokes Codex non-interactively
4. resumes the same Codex session across iterations
5. feeds benchmark summaries back into the next optimization turn

See:

- [agent/README.md](agent/README.md)
- [agent/run_agent.py](agent/run_agent.py)

## Local Packaging

Each kernel directory can be packed independently, for example:

```bash
cd dsa_indexer
python3 scripts/pack_solution.py
```

## Pre-Submission Checklist

- update `solution.name` if a new submission version is desired
- keep `author = "KernelMind"` consistent across all kernel directories
- commit and push the final repository state
- create and push a git tag for evaluation
- if the repo is private, grant read access to `flashinfer-bot`
- send the GitHub repo URL to organizers and explicitly mark it as `Full-Agent`

## Notes

- `FAQ.md` and `EVALUATION.md` are preserved from the starter-kit for reference.
- `SUBMISSION_LAYOUT.md` documents how the current top-level layout was assembled.
