#!/usr/bin/env bash
# scripts/ci-gate.sh — one-line STT benchmark CI gate.
#
# Runs the STT benchmark against a saved baseline and blocks the build
# (non-zero exit) when EITHER gate trips:
#   * a fixture regressed   — PASS in the baseline, now FAIL
#   * a fixture was removed  — present in the baseline, gone from the corpus
#
# It is the committed, copy-pasteable wrapper for the iter-137/138 flags:
#   python scripts/run_stt_benchmark.py --diff <baseline> \
#       --fail-on-regression --fail-on-removed
#
# A CI step is then a single call:
#   scripts/ci-gate.sh --baseline baseline.json
# and the exit code drives the pipeline — no jq/grep plumbing.
#
# Usage:
#   scripts/ci-gate.sh [--baseline PATH] [--engine NAME] [--model M] [-- EXTRA...]
#
#   --baseline PATH   baseline JSON from a prior `--format json` run
#                     (default: baseline.json)
#   --engine NAME     STT engine to benchmark (default: faster_whisper)
#   --model M         model repo / size string (default: engine's own default)
#   -- EXTRA...       everything after `--` is forwarded verbatim to the
#                     benchmark (e.g. --device cpu --compute int8 --format json)
#   -h, --help        print this help and exit 0
#
# Exit codes (passed through from run_stt_benchmark.py, except 2 below):
#   0  no regression and no removal — gate passes
#   1  a fixture regressed OR a baseline fixture was removed
#   2  usage error / missing baseline (this wrapper)
#   3  baseline JSON missing or unparseable (from the benchmark)
#
# Overridable for testing / non-standard layouts:
#   PYTHON            python interpreter         (default: python3 or python)
#   BENCHMARK_SCRIPT  path to run_stt_benchmark.py
#                     (default: alongside this script)
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BASELINE="baseline.json"
ENGINE="faster_whisper"
MODEL=""
EXTRA=()

usage() {
  # Print the leading comment block (skip the shebang) as help text.
  sed -n '2,/^set -uo pipefail$/{/^set -uo pipefail$/d;s/^# \{0,1\}//;p}' "${BASH_SOURCE[0]}"
}

while [ $# -gt 0 ]; do
  case "$1" in
    --baseline) BASELINE="${2:-}"; shift 2 ;;
    --baseline=*) BASELINE="${1#*=}"; shift ;;
    --engine) ENGINE="${2:-}"; shift 2 ;;
    --engine=*) ENGINE="${1#*=}"; shift ;;
    --model) MODEL="${2:-}"; shift 2 ;;
    --model=*) MODEL="${1#*=}"; shift ;;
    --) shift; EXTRA+=("$@"); break ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "ci-gate.sh: unknown argument '$1' (use -- to forward extra benchmark flags)" >&2
      exit 2
      ;;
  esac
done

if [ -z "$BASELINE" ]; then
  echo "ci-gate.sh: --baseline requires a path" >&2
  exit 2
fi

# Resolve the python interpreter and benchmark script.
PYTHON="${PYTHON:-}"
if [ -z "$PYTHON" ]; then
  PYTHON="$(command -v python3 || command -v python || true)"
fi
if [ -z "$PYTHON" ]; then
  echo "ci-gate.sh: no python interpreter found (set PYTHON=...)" >&2
  exit 2
fi

BENCHMARK_SCRIPT="${BENCHMARK_SCRIPT:-$SCRIPT_DIR/run_stt_benchmark.py}"
if [ ! -f "$BENCHMARK_SCRIPT" ]; then
  echo "ci-gate.sh: benchmark script not found: $BENCHMARK_SCRIPT" >&2
  exit 2
fi

# Missing baseline is a usage error with actionable guidance — fail fast (exit
# 2) rather than letting the benchmark exit 3, so CI logs say how to fix it.
if [ ! -f "$BASELINE" ]; then
  echo "ci-gate.sh: baseline not found: $BASELINE" >&2
  echo "  Create one from the current corpus with:" >&2
  echo "    $PYTHON $BENCHMARK_SCRIPT --engine $ENGINE --format json > $BASELINE" >&2
  exit 2
fi

CMD=("$PYTHON" "$BENCHMARK_SCRIPT" --engine "$ENGINE")
if [ -n "$MODEL" ]; then
  CMD+=(--model "$MODEL")
fi
CMD+=(--diff "$BASELINE" --fail-on-regression --fail-on-removed)
if [ "${#EXTRA[@]}" -gt 0 ]; then
  CMD+=("${EXTRA[@]}")
fi

"${CMD[@]}"
exit $?
