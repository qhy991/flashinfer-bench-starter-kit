#!/usr/bin/env python3
"""
Run a target benchmark command and collect a compact JSON summary.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def load_latest_report(target_dir: Path) -> dict | None:
    candidates = [
        target_dir / "_bench_report.json",
        target_dir / "solution.json"
    ]
    report_path = target_dir / "_bench_report.json"
    if report_path.exists():
        try:
            return json.loads(report_path.read_text())
        except json.JSONDecodeError:
            return None
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Run benchmark and capture summary")
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--command", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    proc = subprocess.run(
        args.command,
        cwd=args.target_dir,
        shell=True,
        text=True,
        capture_output=True,
    )

    report = load_latest_report(args.target_dir)
    payload = {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "report": report,
    }
    args.output.write_text(json.dumps(payload, indent=2))
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
