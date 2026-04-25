# Final Submission Checklist

Repository:

- Path: `/home/qinhaiyan/flashinfer-bench-full-agent-submission`
- Submission type: `Full-Agent`
- Team / author: `KernelMind`

## Current Kernel Directories

- `dsa_attention/`
- `dsa_indexer/`
- `gdn_decode/`
- `gdn_prefill/`
- `moe/`

## Validation Status

Verified locally:

- `dsa_attention`
  - full benchmark passed (`23/23`)
- `dsa_indexer`
  - full benchmark passed (`128/128`)
- `gdn_decode`
  - smoke benchmark passed (`2/2`)
- `gdn_prefill`
  - smoke benchmark passed (`2/2`)

## Required Pre-Push Checks

- confirm `solution.name` and `author` are correct in all subdirectories
- confirm root has no `config.toml`
- confirm each top-level kernel directory has its own `config.toml` and `solution/`
- confirm `agent/` exists and documents the full-agent workflow

## Recommended Git Commands

```bash
cd /home/qinhaiyan/flashinfer-bench-full-agent-submission

git add -A
git commit -m "Prepare KernelMind full-agent FlashInfer submission"
git push origin <branch>

git tag submission-v1
git push origin submission-v1
```

If you make a final post-validation update, use a new tag instead of amending:

```bash
git tag submission-v2
git push origin submission-v2
```

## If the Repository Is Private

- grant read access to `flashinfer-bot`

## What to Send to Organizers

Include:

- GitHub repo URL
- team name: `KernelMind`
- explicit note: `This is our Full-Agent submission repository`

Suggested wording:

```text
KernelMind Full-Agent submission repo:
<repo-url>

This repository is our Full-Agent submission for the MLSys 2026 FlashInfer AI Kernel Generation Contest.
```
