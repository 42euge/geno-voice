#!/bin/bash
# E2E integration test runner
#
# Starts voice server + sidecar, runs tests, cleans up.
#
# Usage:
#   ./tests/e2e/run.sh                  # all tests
#   ./tests/e2e/run.sh -k loopback      # only loopback tests
#   ./tests/e2e/run.sh -k "stt"         # only STT tests
#   ./tests/e2e/run.sh --no-loopback    # skip loopback tests (no SwitchAudioSource needed)

set -euo pipefail
cd "$(dirname "$0")/../.."

VENV=".venv/bin/python"
PYTEST_ARGS="${@}"
SKIP_LOOPBACK=false

for arg in "$@"; do
    if [ "$arg" = "--no-loopback" ]; then
        SKIP_LOOPBACK=true
        PYTEST_ARGS="${PYTEST_ARGS/--no-loopback/}"
        PYTEST_ARGS="$PYTEST_ARGS -k 'not Loopback'"
    fi
done

cleanup() {
    echo ""
    echo "Cleaning up..."
    [ -n "${VOICE_PID:-}" ] && kill "$VOICE_PID" 2>/dev/null
    [ -n "${SIDECAR_PID:-}" ] && kill "$SIDECAR_PID" 2>/dev/null
    # Restore audio devices
    SwitchAudioSource -s "MacBook Air Speakers" -t output 2>/dev/null || true
    SwitchAudioSource -s "MacBook Air Microphone" -t input 2>/dev/null || true
    wait 2>/dev/null
}
trap cleanup EXIT

echo "=== MindReflect E2E Tests ==="
echo ""

# Check prerequisites
if ! $VENV -c "import mlx_whisper" 2>/dev/null; then
    echo "ERROR: mlx_whisper not installed"
    exit 1
fi

if ! command -v SwitchAudioSource &>/dev/null && [ "$SKIP_LOOPBACK" = false ]; then
    echo "WARNING: SwitchAudioSource not found. Loopback tests will be skipped."
    echo "Install: brew install switchaudio-osx"
    PYTEST_ARGS="$PYTEST_ARGS -k 'not Loopback'"
fi

# Start voice server
echo "Starting voice server..."
$VENV server.py &
VOICE_PID=$!
until curl -s http://127.0.0.1:5111/health >/dev/null 2>&1; do sleep 1; done
echo "Voice server ready (PID $VOICE_PID)"

# Start sidecar (mic mode for loopback, or test-audio mode)
if [ "$SKIP_LOOPBACK" = false ] && command -v SwitchAudioSource &>/dev/null; then
    echo "Starting sidecar (mic mode for loopback)..."
    SwitchAudioSource -s "Loopback Audio" -t input 2>/dev/null
    $VENV pipecat_server.py &
    SIDECAR_PID=$!
    sleep 3
    echo "Sidecar ready (PID $SIDECAR_PID)"
fi

echo ""
echo "Running tests..."
echo "---"

$VENV -m pytest tests/e2e/ -v --tb=short $PYTEST_ARGS

echo ""
echo "=== Done ==="
