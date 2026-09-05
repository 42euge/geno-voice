#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TEMP_DIR"' EXIT

SOURCES=(
    "$PROJECT_DIR/Sources/GenoQuickAgent/AgentCoordinator.swift"
    "$PROJECT_DIR/Sources/GenoQuickAgent/ClaudeBackendClient.swift"
    "$PROJECT_DIR/Sources/GenoQuickAgent/SpeechRecorder.swift"
)

for optional_source in AgentPhase.swift AppVersion.swift CaptureEngineStore.swift HighlightedAnswerText.swift OverlayLayout.swift OverlayPalette.swift OverlayWindowPolicy.swift PermissionDestination.swift ShortcutPolicy.swift SpeechPlaybackController.swift SpeechPreferences.swift SpokenWordRange.swift; do
    source_path="$PROJECT_DIR/Sources/GenoQuickAgent/$optional_source"
    if [[ -f "$source_path" ]]; then
        SOURCES+=("$source_path")
    fi
done

swiftc \
    -swift-version 5 \
    -framework AppKit \
    -framework AVFoundation \
    -framework Speech \
    -framework SwiftUI \
    "${SOURCES[@]}" \
    "$PROJECT_DIR/Tests/Regression/main.swift" \
    -o "$TEMP_DIR/GenoQuickAgentRegressionTests"

"$TEMP_DIR/GenoQuickAgentRegressionTests"
