"""Tests for scripts/ci-gate.sh — the one-line STT benchmark CI gate.

iter-139: ci-gate.sh is the committed wrapper that wires the iter-137
`--fail-on-regression` and iter-138 `--fail-on-removed` gates end to end so
a CI step is a single call. These tests drive the shell script via subprocess
against a *stub* benchmark (BENCHMARK_SCRIPT override) so they are hermetic —
no real STT engine, model download, or corpus is needed. The stub echoes its
argv (to assert the wrapper forwards the right flags) and exits with a code
taken from the STUB_EXIT env var (to assert the wrapper passes the exit code
through unchanged).
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CI_GATE = REPO_ROOT / "scripts" / "ci-gate.sh"

# A stub that stands in for run_stt_benchmark.py: print the forwarded argv on
# the first line and exit with STUB_EXIT (default 0).
STUB = (
    "import os, sys\n"
    "print('ARGS:', ' '.join(sys.argv[1:]))\n"
    "sys.exit(int(os.environ.get('STUB_EXIT', '0')))\n"
)

bash_required = pytest.mark.skipif(
    shutil.which("bash") is None, reason="bash not available"
)


@pytest.fixture()
def gate_env(tmp_path):
    """Write a stub benchmark + a baseline file, return a runner closure.

    The runner invokes ci-gate.sh with BENCHMARK_SCRIPT pointing at the stub
    and returns the CompletedProcess. `stub_exit` controls the stub's exit
    code so we can assert pass-through.
    """
    stub = tmp_path / "stub.py"
    stub.write_text(STUB)
    baseline = tmp_path / "baseline.json"
    baseline.write_text("{}")

    def run(*args, stub_exit=0, baseline_path=None):
        env = dict(os.environ)
        env["BENCHMARK_SCRIPT"] = str(stub)
        env["STUB_EXIT"] = str(stub_exit)
        bpath = baseline_path if baseline_path is not None else str(baseline)
        cmd = ["bash", str(CI_GATE), "--baseline", bpath, *args]
        return subprocess.run(
            cmd, capture_output=True, text=True, env=env, cwd=str(tmp_path)
        )

    run.tmp_path = tmp_path
    run.baseline = baseline
    return run


@bash_required
def test_script_exists_and_executable():
    assert CI_GATE.is_file()
    assert os.access(CI_GATE, os.X_OK), "ci-gate.sh should be executable"


@bash_required
def test_default_flags_wire_both_gates(gate_env):
    """With no extra args the wrapper passes --diff + both gate flags."""
    res = gate_env(stub_exit=0)
    assert res.returncode == 0
    assert "--engine faster_whisper" in res.stdout
    assert "--diff" in res.stdout
    assert "--fail-on-regression" in res.stdout
    assert "--fail-on-removed" in res.stdout


@bash_required
def test_exit_code_passthrough_clean(gate_env):
    """Benchmark exit 0 (no regression/removal) → gate exits 0."""
    res = gate_env(stub_exit=0)
    assert res.returncode == 0


@bash_required
def test_exit_code_passthrough_blocked(gate_env):
    """Benchmark exit 1 (regression or removal) → gate exits 1."""
    res = gate_env(stub_exit=1)
    assert res.returncode == 1


@bash_required
def test_exit_code_passthrough_baseline_error(gate_env):
    """Benchmark exit 3 (unparseable baseline) propagates unchanged."""
    res = gate_env(stub_exit=3)
    assert res.returncode == 3


@bash_required
def test_engine_override_forwarded(gate_env):
    res = gate_env("--engine", "whisper", stub_exit=0)
    assert "--engine whisper" in res.stdout
    assert "--engine faster_whisper" not in res.stdout


@bash_required
def test_model_forwarded(gate_env):
    res = gate_env("--model", "tiny", stub_exit=0)
    assert "--model tiny" in res.stdout


@bash_required
def test_model_omitted_when_unset(gate_env):
    """No --model flag is forwarded when the caller doesn't set one."""
    res = gate_env(stub_exit=0)
    assert "--model" not in res.stdout


@bash_required
def test_extra_args_forwarded_after_doubledash(gate_env):
    """Everything after `--` reaches the benchmark verbatim."""
    res = gate_env("--", "--device", "cpu", "--compute", "int8", stub_exit=0)
    assert "--device cpu" in res.stdout
    assert "--compute int8" in res.stdout


@bash_required
def test_equals_form_arguments(gate_env):
    """--engine=NAME / --model=M forms work too."""
    res = gate_env("--engine=whisper", "--model=base", stub_exit=0)
    assert "--engine whisper" in res.stdout
    assert "--model base" in res.stdout


@bash_required
def test_gate_flags_precede_extra_args(gate_env):
    """The gate flags are wired before forwarded extras (order sanity)."""
    res = gate_env("--", "--format", "json", stub_exit=0)
    args_line = next(
        line for line in res.stdout.splitlines() if line.startswith("ARGS:")
    )
    assert args_line.index("--fail-on-removed") < args_line.index("--format")


@bash_required
def test_missing_baseline_is_usage_error(gate_env):
    """A baseline path that doesn't exist → exit 2 with creation hint."""
    res = gate_env(stub_exit=0, baseline_path=str(gate_env.tmp_path / "nope.json"))
    assert res.returncode == 2
    assert "baseline not found" in res.stderr
    assert "--format json" in res.stderr  # actionable: how to create one


@bash_required
def test_unknown_argument_is_usage_error(gate_env):
    res = gate_env("--bogus", stub_exit=0)
    assert res.returncode == 2
    assert "unknown argument" in res.stderr


@bash_required
def test_help_exits_zero_and_describes_gates(gate_env):
    res = subprocess.run(
        ["bash", str(CI_GATE), "--help"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "fail-on-regression" in res.stdout
    assert "fail-on-removed" in res.stdout


@bash_required
def test_empty_baseline_value_is_usage_error():
    """--baseline with an empty value is rejected (exit 2)."""
    res = subprocess.run(
        ["bash", str(CI_GATE), "--baseline", ""],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 2
    assert "requires a path" in res.stderr
