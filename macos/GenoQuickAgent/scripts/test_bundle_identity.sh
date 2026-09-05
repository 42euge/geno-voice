#!/usr/bin/env bash
set -euo pipefail

APP_PATH="${1:?usage: test_bundle_identity.sh APP_PATH}"
EXPECTED='designated => identifier "com.geno.quickagent"'
REQUIREMENT="$(codesign -d -r- "$APP_PATH" 2>&1)"

if [[ "$REQUIREMENT" != *"$EXPECTED"* ]]; then
    echo "FAIL: expected stable signing requirement: $EXPECTED" >&2
    echo "$REQUIREMENT" >&2
    exit 1
fi

echo "Bundle identity is stable: $EXPECTED"
