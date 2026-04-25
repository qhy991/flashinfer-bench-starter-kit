# Submission Layout

This folder is a clean submission staging repo based on a fresh clone of the official
`flashinfer-bench-starter-kit`.

## Current Contents

- `dsa_attention/`
  - Best current attention implementation copied from:
    `flashinfer-bench-starter-kit/ako_workspaces/dsa_attention_v8`
  - Includes:
    - `config.toml`
    - `solution/cuda/kernel.cu`
    - `solution/cuda/binding.cpp`
    - local helper scripts in `scripts/`

- `dsa_indexer/`
  - Best current indexer implementation copied from:
    `flashinfer-bench-starter-kit/ako_workspaces/dsa_indexer_v38`
  - Includes:
    - `config.toml`
    - `solution/python/kernel.py`
    - local helper scripts in `scripts/`

- `gdn_decode/`
  - Copied from:
    `openavo_gdn_collection_decode_prefill.zip`
  - Includes:
    - `config.toml`
    - `solution/python/kernel.py`
    - local helper scripts in `scripts/`

- `gdn_prefill/`
  - Copied from:
    `openavo_gdn_collection_decode_prefill.zip`
  - Includes:
    - `config.toml`
    - `solution/python/kernel.py`
    - local helper scripts in `scripts/`

- `moe/`
  - Restored from:
    `moe_codex_20260425_m26a_safe_weakref_tgt1_cache.json`
  - Includes:
    - `config.toml`
    - `solution/python/main.py`
    - local helper scripts in `scripts/`

## Important Notes

- Root-level `config.toml` has been removed on purpose.
  This avoids the evaluation pipeline treating the repository root as an extra submission.

- The repository root still contains the official starter-kit docs and assets.

- Each kernel lives in a top-level subdirectory so the repo can later hold multiple
  definitions for the same submission approach.

## Next Steps

- Update each subdirectory's `config.toml`:
  - `solution.name`
  - `solution.author`

- Add your `agent/` directory and full-agent reproduction materials before submission.
