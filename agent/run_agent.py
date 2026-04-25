#!/usr/bin/env python3
"""
Minimal Codex orchestration loop for FlashInfer full-agent submissions.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import textwrap
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
AGENT_ROOT = REPO_ROOT / "agent"
TARGETS_PATH = AGENT_ROOT / "configs" / "targets.json"
SYSTEM_PROMPT_PATH = AGENT_ROOT / "prompts" / "system_prompt.md"
CONTINUE_PROMPT_PATH = AGENT_ROOT / "prompts" / "continue_prompt.md"


def load_targets() -> dict:
    return json.loads(TARGETS_PATH.read_text())


def read_text(path: Path) -> str:
    return path.read_text().strip()


def timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def copy_target_dir(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def copy_support_dirs(run_root: Path) -> None:
    tools_src = REPO_ROOT / "tools"
    tools_dst = run_root / "tools"
    if tools_src.exists():
        if tools_dst.exists():
            shutil.rmtree(tools_dst)
        shutil.copytree(tools_src, tools_dst)

    scripts_src = REPO_ROOT / "scripts"
    scripts_dst = run_root / "scripts"
    if scripts_src.exists():
        if scripts_dst.exists():
            shutil.rmtree(scripts_dst)
        shutil.copytree(scripts_src, scripts_dst)


def summarize_benchmark(bench_json: dict) -> str:
    report = bench_json.get("report") or {}
    lines = [
        f"returncode: {bench_json.get('returncode')}",
    ]
    if report:
        lines.extend(
            [
                f"compiled: {report.get('compiled')}",
                f"correct: {report.get('correct')}",
                f"passed_workloads: {report.get('passed_workloads')}/{report.get('workload_count')}",
                f"runtime_mean_ms: {report.get('runtime_mean_ms')}",
                f"speedup_mean: {report.get('speedup_mean')}",
                f"status_counts: {report.get('status_counts')}"
            ]
        )
        failures = [row for row in report.get("rows", []) if row.get("status") != "PASSED"][:3]
        for idx, failure in enumerate(failures, start=1):
            lines.append(f"failure_{idx}: {failure}")
    else:
        stdout_tail = (bench_json.get("stdout") or "")[-4000:]
        stderr_tail = (bench_json.get("stderr") or "")[-4000:]
        if stdout_tail:
            lines.append(f"stdout_tail:\n{stdout_tail}")
        if stderr_tail:
            lines.append(f"stderr_tail:\n{stderr_tail}")
    return "\n".join(lines)


def print_iteration_status(iter_idx: int, bench: dict, codex_returncode: int | None = None) -> None:
    report = bench.get("report") or {}
    print(f"[iter {iter_idx}] benchmark returncode={bench.get('returncode')}")
    if report:
        print(
            f"[iter {iter_idx}] compiled={report.get('compiled')} "
            f"correct={report.get('correct')} "
            f"passed={report.get('passed_workloads')}/{report.get('workload_count')} "
            f"runtime_mean_ms={report.get('runtime_mean_ms')} "
            f"speedup_mean={report.get('speedup_mean')}"
        )
    else:
        stderr_tail = (bench.get("stderr") or "").strip()
        stdout_tail = (bench.get("stdout") or "").strip()
        if stderr_tail:
            print(f"[iter {iter_idx}] stderr: {stderr_tail}")
        elif stdout_tail:
            print(f"[iter {iter_idx}] stdout: {stdout_tail[-1000:]}")
    if codex_returncode is not None:
        print(f"[iter {iter_idx}] codex returncode={codex_returncode}")


def print_codex_error(iter_idx: int, proc: subprocess.CompletedProcess) -> None:
    if proc.returncode == 0:
        return
    stderr = (proc.stderr or "").strip()
    stdout = (proc.stdout or "").strip()
    if stderr:
        print(f"[iter {iter_idx}] codex stderr: {stderr[-4000:]}")
    elif stdout:
        print(f"[iter {iter_idx}] codex stdout: {stdout[-4000:]}")


def run_benchmark(target_dir: Path, benchmark_cmd: str, out_dir: Path, iter_idx: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / f"bench_{iter_idx:02d}.json"
    cmd = [
        "python3",
        str(AGENT_ROOT / "tools" / "run_benchmark.py"),
        "--target-dir",
        str(target_dir),
        "--command",
        benchmark_cmd,
        "--output",
        str(output),
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=False)
    return json.loads(output.read_text())


def build_initial_prompt(target_name: str, target_cfg: dict, bench_summary: str) -> str:
    return textwrap.dedent(
        f"""
        {read_text(SYSTEM_PROMPT_PATH)}

        Target: {target_name}
        Definition: {target_cfg['definition']}
        Working directory: {target_cfg['path']}

        Latest benchmark summary:
        {bench_summary}

        Optimize this target. Make one coherent code change aimed at improving
        performance while preserving correctness. Stop after the code change is
        complete.
        """
    ).strip()


def build_continue_prompt(bench_summary: str) -> str:
    return textwrap.dedent(
        f"""
        {read_text(CONTINUE_PROMPT_PATH)}

        Latest benchmark summary:
        {bench_summary}
        """
    ).strip()


def run_codex_exec(work_dir: Path, prompt: str, out_dir: Path, turn_name: str) -> subprocess.CompletedProcess:
    out_dir.mkdir(parents=True, exist_ok=True)
    transcript = out_dir / f"{turn_name}.jsonl"
    last_message = out_dir / f"{turn_name}.last.txt"
    cmd = [
        "codex",
        "exec",
        "--json",
        "--skip-git-repo-check",
        "--dangerously-bypass-approvals-and-sandbox",
        "-C",
        str(work_dir),
        "-o",
        str(last_message),
        prompt,
    ]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True)
    transcript.write_text(proc.stdout)
    return proc


def run_codex_resume(work_dir: Path, prompt: str, out_dir: Path, turn_name: str) -> subprocess.CompletedProcess:
    out_dir.mkdir(parents=True, exist_ok=True)
    transcript = out_dir / f"{turn_name}.jsonl"
    last_message = out_dir / f"{turn_name}.last.txt"
    cmd = [
        "codex",
        "exec",
        "resume",
        "--last",
        "--json",
        "--skip-git-repo-check",
        "--dangerously-bypass-approvals-and-sandbox",
        "-o",
        str(last_message),
        prompt,
    ]
    proc = subprocess.run(cmd, cwd=work_dir, text=True, capture_output=True)
    transcript.write_text(proc.stdout)
    return proc


def write_state(path: Path, state: dict) -> None:
    path.write_text(json.dumps(state, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Codex full-agent optimization loop")
    parser.add_argument("--target", required=True, help="Target key from agent/configs/targets.json")
    parser.add_argument("--max-iters", type=int, default=3)
    parser.add_argument("--work-root", type=Path, default=REPO_ROOT / "agent_runs")
    args = parser.parse_args()

    targets = load_targets()
    if args.target not in targets:
        raise SystemExit(f"Unknown target: {args.target}")

    target_cfg = targets[args.target]
    src_target_dir = REPO_ROOT / target_cfg["path"]
    run_root = args.work_root / f"{args.target}_{timestamp()}"
    work_dir = run_root / target_cfg["path"]
    codex_dir = run_root / "codex"
    bench_dir = run_root / "benchmarks"
    state_path = run_root / "run_state.json"
    run_root.mkdir(parents=True, exist_ok=True)

    copy_target_dir(src_target_dir, work_dir)
    copy_support_dirs(run_root)

    state = {
        "target": args.target,
        "source_target_dir": str(src_target_dir),
        "work_dir": str(work_dir),
        "iterations": [],
    }
    write_state(state_path, state)

    bench = run_benchmark(work_dir, target_cfg["benchmark_cmd"], bench_dir, 0)
    bench_summary = summarize_benchmark(bench)
    proc = run_codex_exec(work_dir, build_initial_prompt(args.target, target_cfg, bench_summary), codex_dir, "iter_00_codex")
    print_iteration_status(0, bench, proc.returncode)
    print_codex_error(0, proc)
    state["iterations"].append(
        {
            "iter": 0,
            "benchmark": bench,
            "codex_returncode": proc.returncode,
            "codex_stdout": proc.stdout,
            "codex_stderr": proc.stderr,
        }
    )
    write_state(state_path, state)

    for iter_idx in range(1, args.max_iters + 1):
        bench = run_benchmark(work_dir, target_cfg["benchmark_cmd"], bench_dir, iter_idx)
        bench_summary = summarize_benchmark(bench)
        proc = run_codex_resume(work_dir, build_continue_prompt(bench_summary), codex_dir, f"iter_{iter_idx:02d}_codex")
        print_iteration_status(iter_idx, bench, proc.returncode)
        print_codex_error(iter_idx, proc)
        state["iterations"].append(
            {
                "iter": iter_idx,
                "benchmark": bench,
                "codex_returncode": proc.returncode,
                "codex_stdout": proc.stdout,
                "codex_stderr": proc.stderr,
            }
        )
        write_state(state_path, state)

        report = bench.get("report") or {}
        if report.get("correct") and iter_idx >= args.max_iters:
            break

    print(json.dumps({"run_root": str(run_root), "state_file": str(state_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
