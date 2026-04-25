You are optimizing a FlashInfer contest kernel inside a starter-kit directory.

Rules:

- Work only inside the current target directory.
- Preserve correctness under the benchmark.
- Do not reward-hack the benchmark.
- Prefer small, testable changes.
- After making changes, rely on the external benchmark loop to validate them.
- If the latest benchmark is already correct and fast, only continue if you have
  a concrete hypothesis.
- Do not manually edit files outside the target directory.

Expected workflow:

1. Read `config.toml` and the files under `solution/`.
2. Inspect the latest benchmark summary provided in the prompt.
3. Make one coherent optimization attempt.
4. Stop after the code changes are complete.
