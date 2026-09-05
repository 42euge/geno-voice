#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_APP_NAME="Geno Quick Agent"
EXECUTABLE_NAME="GenoQuickAgent"
CONFIGURATION="${GENO_QUICK_AGENT_BUILD_CONFIGURATION:-release}"
DIST_DIR="${GENO_QUICK_AGENT_DIST_DIR:-$PROJECT_DIR/dist}"
SIGNING_IDENTITY="${CODESIGN_IDENTITY:--}"
INFO_PLIST="$PROJECT_DIR/Resources/Info.plist"
ENTITLEMENTS_PLIST="$PROJECT_DIR/Resources/GenoQuickAgent.entitlements"
BUNDLE_ID="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleIdentifier' "$INFO_PLIST")"
SHORT_VERSION="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' "$INFO_PLIST")"
BUILD_NUMBER="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleVersion' "$INFO_PLIST")"
VERSION_LABEL="v$SHORT_VERSION ($BUILD_NUMBER)"
VERSIONED_APP_NAME="$BASE_APP_NAME $VERSION_LABEL"
APP_BUNDLE="$DIST_DIR/$VERSIONED_APP_NAME.app"
DMG_PATH="$DIST_DIR/$VERSIONED_APP_NAME.dmg"
LATEST_DMG_PATH="$DIST_DIR/$BASE_APP_NAME.dmg"
LEGACY_APP_BUNDLE="$DIST_DIR/$BASE_APP_NAME.app"

"$PROJECT_DIR/scripts/test_regressions.sh"
node --test "$PROJECT_DIR/Backend/claude-backend.test.mjs"
swift build --package-path "$PROJECT_DIR" -c "$CONFIGURATION"
BIN_DIR="$(swift build --package-path "$PROJECT_DIR" -c "$CONFIGURATION" --show-bin-path)"

if [[ -d "$APP_BUNDLE" ]]; then
    rm -rf "$APP_BUNDLE"
fi
if [[ -d "$LEGACY_APP_BUNDLE" ]]; then
    rm -rf "$LEGACY_APP_BUNDLE"
fi
mkdir -p "$APP_BUNDLE/Contents/MacOS" "$APP_BUNDLE/Contents/Resources" "$DIST_DIR"
cp "$BIN_DIR/$EXECUTABLE_NAME" "$APP_BUNDLE/Contents/MacOS/$EXECUTABLE_NAME"
cp "$INFO_PLIST" "$APP_BUNDLE/Contents/Info.plist"
/usr/libexec/PlistBuddy -c "Set :CFBundleDisplayName $VERSIONED_APP_NAME" "$APP_BUNDLE/Contents/Info.plist"
/usr/libexec/PlistBuddy -c "Set :CFBundleName $VERSIONED_APP_NAME" "$APP_BUNDLE/Contents/Info.plist"
"$PROJECT_DIR/scripts/prepare_claude_backend.sh" \
    "$APP_BUNDLE/Contents/Resources/ClaudeBackend"

TEMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TEMP_DIR"' EXIT
ICONSET_DIR="$TEMP_DIR/AppIcon.iconset"
swift "$PROJECT_DIR/scripts/make_icon.swift" "$ICONSET_DIR"
iconutil -c icns "$ICONSET_DIR" -o "$APP_BUNDLE/Contents/Resources/AppIcon.icns"

if [[ "$SIGNING_IDENTITY" == "-" ]]; then
    codesign \
        --force \
        --deep \
        --options runtime \
        --identifier "$BUNDLE_ID" \
        --entitlements "$ENTITLEMENTS_PLIST" \
        --requirements "=designated => identifier \"$BUNDLE_ID\"" \
        --sign - \
        "$APP_BUNDLE"
    "$PROJECT_DIR/scripts/test_bundle_identity.sh" "$APP_BUNDLE"
else
    codesign \
        --force \
        --deep \
        --options runtime \
        --entitlements "$ENTITLEMENTS_PLIST" \
        --sign "$SIGNING_IDENTITY" \
        "$APP_BUNDLE"
fi
"$PROJECT_DIR/scripts/test_bundle_permissions.sh" "$APP_BUNDLE"
"$PROJECT_DIR/scripts/test_bundle_backend.sh" "$APP_BUNDLE"

DMG_STAGE="$TEMP_DIR/dmg"
mkdir -p "$DMG_STAGE"
cp -R "$APP_BUNDLE" "$DMG_STAGE/"
ln -s /Applications "$DMG_STAGE/Applications"
if [[ -f "$DMG_PATH" ]]; then
    rm -f "$DMG_PATH"
fi
hdiutil create \
    -volname "$VERSIONED_APP_NAME" \
    -srcfolder "$DMG_STAGE" \
    -format UDZO \
    -ov \
    "$DMG_PATH"
cp "$DMG_PATH" "$LATEST_DMG_PATH"

echo "Built $APP_BUNDLE"
echo "Built $DMG_PATH"
echo "Updated $LATEST_DMG_PATH"
