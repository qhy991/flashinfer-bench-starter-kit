#!/bin/bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

LABEL=""
if [ $# -gt 0 ] && [[ "$1" != --* ]]; then
    LABEL="$1"
    shift
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

set +e
python3 ../tools/ako_flashinfer_bench.py bench --workspace "$ROOT_DIR" "$@" 2>&1 | tee _bench_output.txt
BENCH_EXIT=$?
set -e

if [ -n "$LABEL" ]; then
    TRAJ_DIR="trajectory/${TIMESTAMP}_${LABEL}"
else
    TRAJ_DIR="trajectory/${TIMESTAMP}"
fi

mkdir -p "$TRAJ_DIR"
if [ -d solution ]; then
    cp -r solution/. "$TRAJ_DIR/"
fi
[ -f _bench_output.txt ] && mv _bench_output.txt "$TRAJ_DIR/output.txt"
[ -f _bench_report.json ] && mv _bench_report.json "$TRAJ_DIR/report.json"

echo "Trajectory saved to: $TRAJ_DIR"
exit $BENCH_EXIT
