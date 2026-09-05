#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${GENO_QUICK_AGENT_PROTOTYPE_PORT:-4173}"

echo "GenoVoice overlay prototype: http://127.0.0.1:${PORT}/?variant=A"
python3 -m http.server "$PORT" --bind 127.0.0.1 --directory "$REPO_DIR/prototypes/quick-agent-overlay"
