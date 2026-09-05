#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:?usage: prepare_claude_backend.sh OUTPUT_DIR}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_DIR="$PROJECT_DIR/Backend"
CACHE_DIR="${GENO_QUICK_AGENT_BACKEND_CACHE_DIR:-$PROJECT_DIR/.build/claude-backend-cache}"
NODE_VERSION="22.22.3"
NODE_PACKAGE="node-v$NODE_VERSION-darwin-arm64"
NODE_ARCHIVE="$CACHE_DIR/$NODE_PACKAGE.tar.gz"
NODE_DIRECTORY="$CACHE_DIR/$NODE_PACKAGE"
NODE_URL="https://nodejs.org/dist/v$NODE_VERSION/$NODE_PACKAGE.tar.gz"
NODE_SHA256="0da7ff74ef8611328c8212f17943368713a2ad953fb7d89a8c8a0eae87c23207"

mkdir -p "$CACHE_DIR"
if [[ ! -f "$NODE_ARCHIVE" ]]; then
    curl --fail --location --silent --show-error "$NODE_URL" --output "$NODE_ARCHIVE"
fi
printf '%s  %s\n' "$NODE_SHA256" "$NODE_ARCHIVE" | shasum -a 256 --check --status

if [[ ! -x "$NODE_DIRECTORY/bin/node" ]]; then
    if [[ -d "$NODE_DIRECTORY" ]]; then
        rm -rf "$NODE_DIRECTORY"
    fi
    tar -xzf "$NODE_ARCHIVE" -C "$CACHE_DIR"
fi

NODE_DESCRIPTION="$(file "$NODE_DIRECTORY/bin/node")"
if [[ "$NODE_DESCRIPTION" != *"arm64"* ]]; then
    echo "Bundled Node runtime is not arm64: $NODE_DESCRIPTION" >&2
    exit 1
fi

npm --prefix "$BACKEND_DIR" ci --omit=dev --no-audit --ignore-scripts --prefer-offline

mkdir -p "$OUTPUT_DIR"
cp "$NODE_DIRECTORY/bin/node" "$OUTPUT_DIR/node"
cp "$BACKEND_DIR/claude-backend.mjs" "$OUTPUT_DIR/claude-backend.mjs"
cp "$BACKEND_DIR/package.json" "$OUTPUT_DIR/package.json"
cp "$BACKEND_DIR/package-lock.json" "$OUTPUT_DIR/package-lock.json"
if [[ -d "$OUTPUT_DIR/node_modules" ]]; then
    rm -rf "$OUTPUT_DIR/node_modules"
fi
ditto "$BACKEND_DIR/node_modules" "$OUTPUT_DIR/node_modules"
chmod +x "$OUTPUT_DIR/node"

echo "Prepared self-contained Claude backend in $OUTPUT_DIR"
