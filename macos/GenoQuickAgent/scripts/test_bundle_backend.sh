#!/usr/bin/env bash
set -euo pipefail

APP_PATH="${1:?usage: test_bundle_backend.sh APP_PATH}"
BACKEND_PATH="$APP_PATH/Contents/Resources/ClaudeBackend"

require_file() {
    local path="$1"
    if [[ ! -f "$path" ]]; then
        echo "FAIL: missing bundled backend file: $path" >&2
        exit 1
    fi
}

require_executable() {
    local path="$1"
    require_file "$path"
    if [[ ! -x "$path" ]]; then
        echo "FAIL: bundled backend file is not executable: $path" >&2
        exit 1
    fi
}

require_executable "$BACKEND_PATH/node"
require_file "$BACKEND_PATH/claude-backend.mjs"
require_file "$BACKEND_PATH/node_modules/@anthropic-ai/claude-agent-sdk/package.json"
require_executable \
    "$BACKEND_PATH/node_modules/@anthropic-ai/claude-agent-sdk-darwin-arm64/claude"

NODE_ARCH="$(file "$BACKEND_PATH/node")"
if [[ "$NODE_ARCH" != *"arm64"* ]]; then
    echo "FAIL: bundled Node runtime is not arm64: $NODE_ARCH" >&2
    exit 1
fi

NODE_VERSION="$("$BACKEND_PATH/node" --version)"
if [[ "$NODE_VERSION" != v22.* ]]; then
    echo "FAIL: expected bundled Node 22, got $NODE_VERSION" >&2
    exit 1
fi

echo "Claude backend bundle is self-contained ($NODE_VERSION, arm64)."
