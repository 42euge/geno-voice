#!/usr/bin/env bash
set -euo pipefail

APP_PATH="${1:?usage: test_bundle_permissions.sh APP_PATH}"
TEMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TEMP_DIR"' EXIT
ENTITLEMENTS_PATH="$TEMP_DIR/entitlements.plist"

codesign -d --entitlements :- "$APP_PATH" >"$ENTITLEMENTS_PATH" 2>/dev/null

if [[ ! -s "$ENTITLEMENTS_PATH" ]]; then
    echo "FAIL: signed app has no entitlements" >&2
    exit 1
fi

AUDIO_INPUT="$(
    /usr/libexec/PlistBuddy \
        -c 'Print :com.apple.security.device.audio-input' \
        "$ENTITLEMENTS_PATH" 2>/dev/null || true
)"
if [[ "$AUDIO_INPUT" != "true" ]]; then
    echo "FAIL: com.apple.security.device.audio-input is not enabled" >&2
    exit 1
fi

echo "Bundle microphone entitlement is enabled."
