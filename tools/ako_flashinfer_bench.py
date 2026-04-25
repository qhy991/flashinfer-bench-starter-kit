#!/usr/bin/env python3
"""
AKO-style benchmark bridge for FlashInfer-Bench workspaces.

This script packs a workspace-local solution and runs it through the official
flashinfer_bench benchmark pipeline for a single definition.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
PACK_SCRIPT = REPO_ROOT / "scripts" / "pack_solution.py"


def _ensure_cuda_home() -> None:
    if os.environ.get("CUDA_HOME"):
        return

    for candidate in ("/usr/local/cuda-12.2", "/usr/local/cuda"):
        if os.path.isdir(candidate):
            os.environ["CUDA_HOME"] = candidate
            os.environ["CUDA_PATH"] = candidate
            return


_ensure_cuda_home()

from flashinfer_bench import Benchmark, BenchmarkConfig, Solution, TraceSet


def load_pack_module():
    spec = importlib.util.spec_from_file_location("fib_pack_solution", PACK_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def get_trace_set_path() -> str:
    path = os.environ.get("FIB_DATASET_PATH")
    if not path:
        raise EnvironmentError(
            "FIB_DATASET_PATH is not set. Point it at the FlashInfer trace dataset root."
        )
    return path


def pack_workspace_solution(workspace: Path) -> Path:
    pack_module = load_pack_module()
    pack_module.PROJECT_ROOT = workspace
    return pack_module.pack_solution(workspace / "solution.json")


def build_report(
    workspace: Path,
    solution: Solution,
    result_trace_set,
    definition_name: str,
    workload_count: int,
) -> dict:
    rows = []
    for trace in result_trace_set.traces.get(definition_name, []):
        if not trace.evaluation:
            continue

        row = {
            "workload": trace.workload.uuid,
            "status": trace.evaluation.status.value,
            "solution": trace.solution,
        }
        if trace.evaluation.log:
            row["log"] = trace.evaluation.log
        if trace.evaluation.performance:
            row["latency_ms"] = trace.evaluation.performance.latency_ms
            row["reference_latency_ms"] = trace.evaluation.performance.reference_latency_ms
            row["speedup_factor"] = trace.evaluation.performance.speedup_factor
        if trace.evaluation.correctness:
            row["max_abs_error"] = trace.evaluation.correctness.max_absolute_error
            row["max_rel_error"] = trace.evaluation.correctness.max_relative_error
        rows.append(row)

    status_counts = Counter(row["status"] for row in rows)
    passed = [row for row in rows if row["status"] == "PASSED"]
    compiled = status_counts.get("COMPILE_ERROR", 0) == 0
    correct = len(passed) == workload_count and workload_count > 0

    latencies = [row["latency_ms"] for row in passed if "latency_ms" in row]
    ref_latencies = [row["reference_latency_ms"] for row in passed if "reference_latency_ms" in row]
    speedups = [row["speedup_factor"] for row in passed if "speedup_factor" in row]

    return {
        "workspace": str(workspace),
        "solution": solution.name,
        "definition": solution.definition,
        "compiled": compiled,
        "correct": correct,
        "workload_count": workload_count,
        "passed_workloads": len(passed),
        "status_counts": dict(sorted(status_counts.items())),
        "runtime_mean_ms": sum(latencies) / len(latencies) if latencies else None,
        "runtime_min_ms": min(latencies) if latencies else None,
        "runtime_max_ms": max(latencies) if latencies else None,
        "reference_runtime_mean_ms": sum(ref_latencies) / len(ref_latencies) if ref_latencies else None,
        "speedup_mean": sum(speedups) / len(speedups) if speedups else None,
        "speedup_min": min(speedups) if speedups else None,
        "speedup_max": max(speedups) if speedups else None,
        "rows": rows,
    }


def print_summary(report: dict, report_path: Path) -> None:
    print(f"WORKSPACE: {report['workspace']}")
    print(f"SOLUTION: {report['solution']}")
    print(f"DEFINITION: {report['definition']}")
    print(f"COMPILED: {report['compiled']}")
    print(f"CORRECT: {report['correct']}")
    print(f"PASSED_WORKLOADS: {report['passed_workloads']}/{report['workload_count']}")

    status_counts = report["status_counts"]
    if status_counts:
        status_line = ", ".join(f"{key}={value}" for key, value in status_counts.items())
    else:
        status_line = "none"
    print(f"STATUS_COUNTS: {status_line}")

    for key in (
        "runtime_mean_ms",
        "runtime_min_ms",
        "runtime_max_ms",
        "reference_runtime_mean_ms",
        "speedup_mean",
        "speedup_min",
        "speedup_max",
    ):
        value = report[key]
        if value is None:
            print(f"{key.upper()}: n/a")
        else:
            print(f"{key.upper()}: {value:.6f}")

    failures = [row for row in report["rows"] if row["status"] != "PASSED"][:2]
    for idx, row in enumerate(failures, start=1):
        print(f"FAILURE_{idx}_WORKLOAD: {row['workload']}")
        print(f"FAILURE_{idx}_STATUS: {row['status']}")
        if "max_abs_error" in row:
            print(f"FAILURE_{idx}_MAX_ABS_ERROR: {row['max_abs_error']}")
        if "max_rel_error" in row:
            print(f"FAILURE_{idx}_MAX_REL_ERROR: {row['max_rel_error']}")
        if "log" in row and row["log"].strip():
            log = row["log"].strip().replace("\n", " | ")
            print(f"FAILURE_{idx}_LOG: {log[:1500]}")

    print(f"REPORT: {report_path}")


def run_benchmark(args: argparse.Namespace) -> int:
    workspace = args.workspace.resolve()
    solution_path = pack_workspace_solution(workspace)
    solution = Solution.model_validate_json(solution_path.read_text())

    trace_set = TraceSet.from_path(get_trace_set_path())
    if solution.definition not in trace_set.definitions:
        raise ValueError(f"Definition '{solution.definition}' not found in trace set")

    definition = trace_set.definitions[solution.definition]
    workloads = trace_set.workloads.get(solution.definition, [])
    if not workloads:
        raise ValueError(f"No workloads found for definition '{solution.definition}'")

    if args.max_workloads > 0:
        workloads = workloads[: args.max_workloads]

    bench_trace_set = TraceSet(
        root=trace_set.root,
        definitions={definition.name: definition},
        solutions={definition.name: [solution]},
        workloads={definition.name: workloads},
        traces={definition.name: []},
    )

    config = BenchmarkConfig(
        warmup_runs=args.warmup,
        iterations=args.iterations,
        num_trials=args.num_trials,
    )
    result_trace_set = Benchmark(bench_trace_set, config).run_all(dump_traces=True)

    report = build_report(
        workspace=workspace,
        solution=solution,
        result_trace_set=result_trace_set,
        definition_name=definition.name,
        workload_count=len(workloads),
    )
    report_path = workspace / "_bench_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print_summary(report, report_path)
    return 0 if report["correct"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AKO-style FlashInfer benchmark bridge")
    subparsers = parser.add_subparsers(dest="command", required=True)

    bench = subparsers.add_parser("bench", help="Pack and benchmark a workspace-local solution")
    bench.add_argument("--workspace", type=Path, required=True, help="Workspace root directory")
    bench.add_argument("--warmup", type=int, default=3, help="Warmup runs per workload")
    bench.add_argument("--iterations", type=int, default=100, help="Timing iterations per trial")
    bench.add_argument("--num-trials", type=int, default=3, help="Number of trials per workload")
    bench.add_argument("--max-workloads", type=int, default=0, help="Limit workload count (0 = all)")
    bench.set_defaults(func=run_benchmark)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return args.func(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
