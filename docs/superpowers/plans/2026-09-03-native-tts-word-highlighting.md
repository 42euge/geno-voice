# Native TTS and Word Highlighting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add configurable native text-to-speech, synchronized spoken-word highlighting, and a stable dark answer card to GenoVoice.

**Architecture:** A deep `SpeechPlaybackController` module owns `AVSpeechSynthesizer`, preferences, delegate timing, and cancellation behind a small playback interface. `AgentCoordinator` publishes only speaking state and the active UTF-16 word range; SwiftUI renders that range and never sees the synthesizer. Explicit near-black palette colors replace adaptive material.

**Tech Stack:** Swift 5.10, AppKit, SwiftUI, AVFoundation `AVSpeechSynthesizer`, Foundation `UserDefaults`, existing shell/Swift regression harness.

## Global Constraints

- macOS 13 or newer on Apple Silicon.
- Bundle identifier remains `com.geno.quickagent`.
- Native speech synthesis must make no network request and require no new permission.
- Default settings are auto-speak on, system voice, and rate `0.48` clamped to `0.35...0.60`.
- Dismiss, new question, and error transitions stop playback and clear highlighting.
- Existing Option–Space dismissal and compact/off-screen overlay behavior must not regress.
- Release version is `v0.2.1 (10)`.

---

### Task 1: Speech preferences and UTF-16 word-range mapping

**Files:**
- Create: `macos/GenoQuickAgent/Sources/GenoQuickAgent/SpeechPreferences.swift`
- Create: `macos/GenoQuickAgent/Sources/GenoQuickAgent/SpokenWordRange.swift`
- Modify: `macos/GenoQuickAgent/Tests/Regression/main.swift`
- Modify: `macos/GenoQuickAgent/scripts/test_regressions.sh`

**Interfaces:**
- Produces: `SpeechPreferences.registerDefaults(in:)`, `SpeechPreferences.settings(in:)`, `SpeechPlaybackSettings`, and `SpokenWordRange.range(_:in:)`.
- Consumes: Foundation `UserDefaults` and `NSRange` only.

- [ ] **Step 1: Add failing preference and range regressions**

Append tests that create an isolated `UserDefaults` suite, register defaults,
and assert auto-speak `true`, empty voice identifier, rate `0.48`, lower clamp
`0.35`, and upper clamp `0.60`. Add literal UTF-16 cases asserting an emoji
prefix maps the spoken `NSRange` to the intended word and invalid ranges return
`nil`.

```swift
let suiteName = "GenoQuickAgentSpeechPreferences-\(UUID().uuidString)"
let defaults = UserDefaults(suiteName: suiteName)!
defer { defaults.removePersistentDomain(forName: suiteName) }
SpeechPreferences.registerDefaults(in: defaults)
expect(SpeechPreferences.settings(in: defaults).autoSpeak, "auto-speak defaults on")

let emojiText = "Hi 👋 there"
let thereRange = (emojiText as NSString).range(of: "there")
expect(
    SpokenWordRange.substring(thereRange, in: emojiText) == "there",
    "UTF-16 delegate range should survive an emoji prefix"
)
```

- [ ] **Step 2: Run the regression suite and verify RED**

Run: `cd macos/GenoQuickAgent && ./scripts/test_regressions.sh`

Expected: compile failure because `SpeechPreferences` and `SpokenWordRange` do
not exist.

- [ ] **Step 3: Implement minimal preference and range modules**

`SpeechPlaybackSettings` contains `autoSpeak`, `voiceIdentifier`, and `rate`.
`SpeechPreferences` defines the three stable keys, registers defaults, clamps
the stored rate, and returns a settings value. `SpokenWordRange` uses
`Range(nsRange, in: text)` so UTF-16 delegate offsets are converted safely.

```swift
enum SpokenWordRange {
    static func range(_ range: NSRange?, in text: String) -> Range<String.Index>? {
        guard let range, range.location != NSNotFound, range.length > 0 else { return nil }
        return Range(range, in: text)
    }

    static func substring(_ range: NSRange?, in text: String) -> String? {
        guard let converted = range(range, in: text) else { return nil }
        return String(text[converted])
    }
}
```

- [ ] **Step 4: Add both sources to the regression compiler and verify GREEN**

Run: `cd macos/GenoQuickAgent && ./scripts/test_regressions.sh`

Expected: all GenoVoice regression tests pass.

---

### Task 2: Native speech playback and coordinator lifecycle

**Files:**
- Create: `macos/GenoQuickAgent/Sources/GenoQuickAgent/SpeechPlaybackController.swift`
- Modify: `macos/GenoQuickAgent/Sources/GenoQuickAgent/AgentCoordinator.swift`
- Modify: `macos/GenoQuickAgent/Sources/GenoQuickAgent/AppDelegate.swift`
- Modify: `macos/GenoQuickAgent/Tests/Regression/main.swift`
- Modify: `macos/GenoQuickAgent/scripts/test_regressions.sh`

**Interfaces:**
- Consumes: `SpeechPreferences.settings(in:)` and completed answer text.
- Produces: `SpeechPlaybackControlling.playAutomatically(_:)`, `toggle(_:)`, `stop()`, `onSpeakingChanged`, and `onSpokenRangeChanged`; coordinator properties `isSpeaking`, `spokenRange`, and `toggleAnswerPlayback()`.

- [ ] **Step 1: Add a fake playback adapter and failing coordinator regressions**

The fake records automatic-play, toggle, and stop calls and exposes the two
callbacks. Inject it into `AgentCoordinator`. Tests assert:

- receiving an answer invokes automatic playback and enters `.answer`;
- delegate callbacks publish speaking state and active range;
- `toggleAnswerPlayback()` passes the current answer;
- `cancelAndDismiss()`, `startQuestion()`, and `presentError()` call `stop()` and
  clear speaking state/range.

```swift
@MainActor
private final class FakeSpeechPlayback: SpeechPlaybackControlling {
    var onSpeakingChanged: ((Bool) -> Void)?
    var onSpokenRangeChanged: ((NSRange?) -> Void)?
    var automaticTexts: [String] = []
    var toggledTexts: [String] = []
    private(set) var stopCount = 0

    func playAutomatically(_ text: String) { automaticTexts.append(text) }
    func toggle(_ text: String) { toggledTexts.append(text) }
    func stop() { stopCount += 1 }
}
```

- [ ] **Step 2: Run regressions and verify RED**

Run: `cd macos/GenoQuickAgent && ./scripts/test_regressions.sh`

Expected: compile failure because `SpeechPlaybackControlling` and the new
coordinator interface do not exist.

- [ ] **Step 3: Implement the native playback module**

Create a `@MainActor` protocol and an `NSObject` controller wrapping one
`AVSpeechSynthesizer`. `playAutomatically` reads current settings and returns
without speaking when disabled. `toggle` stops active speech or starts from the
beginning regardless of auto-speak. Utterances use the configured installed
voice when available, otherwise `AVSpeechSynthesisVoice(language: "en-US")`,
and use the clamped rate.

Delegate callbacks publish speaking state and `willSpeakRangeOfSpeechString`.
Finish and cancellation callbacks clear both values. Replacing an utterance
always stops the previous one first.

- [ ] **Step 4: Integrate playback into the coordinator lifecycle**

Add injected playback with a native default, wire callbacks to published state,
and centralize cleanup in `stopAnswerPlayback()`. Add internal
`receiveAnswer(_:)`, called by the Claude task and by regressions, to set the
answer/phase and start automatic playback. Register preference defaults during
app launch.

- [ ] **Step 5: Add the playback source to the regression compiler and verify GREEN**

Run: `cd macos/GenoQuickAgent && ./scripts/test_regressions.sh`

Expected: all regressions pass with the fake adapter and no audio hardware
dependency.

---

### Task 3: Highlighted answer, dark surface, and settings controls

**Files:**
- Create: `macos/GenoQuickAgent/Sources/GenoQuickAgent/HighlightedAnswerText.swift`
- Modify: `macos/GenoQuickAgent/Sources/GenoQuickAgent/OverlayView.swift`
- Modify: `macos/GenoQuickAgent/Sources/GenoQuickAgent/SettingsView.swift`
- Modify: `macos/GenoQuickAgent/Sources/GenoQuickAgent/SettingsWindowController.swift`
- Modify: `macos/GenoQuickAgent/Tests/Regression/main.swift`
- Modify: `macos/GenoQuickAgent/scripts/test_regressions.sh`

**Interfaces:**
- Consumes: answer text, `AgentCoordinator.spokenRange`, `isSpeaking`, and the three `SpeechPreferences` keys.
- Produces: attributed answer text with one emphasized range, Speak/Stop action, voice/rate/auto-speak controls, and explicit dark palette colors.

- [ ] **Step 1: Add failing highlighted-text and palette regressions**

Tests assert `HighlightedAnswerText.activeWord(text:range:)` returns the literal
spoken word for valid Unicode ranges and `nil` for stale ranges. Expose
`OverlayPalette.surfaceTop` and `.surfaceBottom` as `NSColor`; assert converted
sRGB red/green/blue components are below `0.16` and alpha is at least `0.96`.

- [ ] **Step 2: Run regressions and verify RED**

Run: `cd macos/GenoQuickAgent && ./scripts/test_regressions.sh`

Expected: compile failure because `HighlightedAnswerText` and `OverlayPalette`
do not exist.

- [ ] **Step 3: Implement attributed word highlighting**

Build an `AttributedString` from the whole answer. Convert the `NSRange` through
`SpokenWordRange`; when valid, convert the Swift range to attributed-string
indices, then apply white foreground, purple background, and semibold font only
to the active word. Invalid ranges return unhighlighted high-contrast text.

- [ ] **Step 4: Replace adaptive material and render playback state**

Replace `.ultraThinMaterial` with an explicit near-black purple gradient from
`OverlayPalette`, apply a dark color scheme to the overlay root, and retain the
border and shadow. Render `HighlightedAnswerText.attributed(...)` in the answer
scroll view. Add an action whose title/icon is `Stop`/`stop.fill` while speaking
and `Speak`/`speaker.wave.2` otherwise.

- [ ] **Step 5: Add basic speech settings**

Use `@AppStorage` for auto-speak, voice identifier, and rate. Add a “Speech”
group containing an auto-speak toggle, “System Default” plus installed English
voices, and a `0.35...0.60` rate slider showing two decimal places. Increase the
fixed Settings view/window height only enough to fit the new group.

- [ ] **Step 6: Verify GREEN and compile the full package**

Run:

```bash
cd macos/GenoQuickAgent
./scripts/test_regressions.sh
swift build -c debug
```

Expected: regressions pass and the app builds without warnings.

---

### Task 4: Version, documentation, packaging, install, and live verification

**Files:**
- Modify: `macos/GenoQuickAgent/Resources/Info.plist`
- Modify: `macos/GenoQuickAgent/Package.swift`
- Modify: `macos/GenoQuickAgent/README.md`

**Interfaces:**
- Consumes: completed native playback and UI changes.
- Produces: signed `GenoVoice v0.2.1 (10).app` and `.dmg`, installed and running.

- [ ] **Step 1: Bump version and remove obsolete linkage**

Set `CFBundleShortVersionString` to `0.2.1` and `CFBundleVersion` to `10`.
Remove the unused Security framework from `Package.swift`; no Keychain source
remains.

- [ ] **Step 2: Update README**

Document native offline TTS, default auto-speak, Settings controls, current-word
highlighting, Speak/Stop, and playback cancellation. Update all artifact names
to `v0.2.1 (10)`.

- [ ] **Step 3: Run full release verification and build the DMG**

Run:

```bash
cd macos/GenoQuickAgent
./scripts/test_regressions.sh
node --test Backend/claude-backend.test.mjs
swift build -c release
./scripts/build_dmg.sh
./scripts/test_bundle_identity.sh "dist/GenoVoice v0.2.1 (10).app"
./scripts/test_bundle_permissions.sh "dist/GenoVoice v0.2.1 (10).app"
./scripts/test_bundle_backend.sh "dist/GenoVoice v0.2.1 (10).app"
codesign --verify --deep --strict "dist/GenoVoice v0.2.1 (10).app"
hdiutil verify "dist/GenoVoice v0.2.1 (10).dmg"
```

Expected: all commands exit zero.

- [ ] **Step 4: Install and launch only the new version**

Quit the running executable, move `/Applications/GenoVoice v0.2.0
(9).app` to Trash, copy the v0.2.1 app to `/Applications`, launch it, and verify
the process remains alive with zero startup windows.

- [ ] **Step 5: Perform live answer-card verification**

Invoke Option–Space, ask a harmless short question, and confirm the answer is
spoken, the current word visibly advances, Speak changes to Stop during
playback, the card is near-black rather than grey, and a second Option–Space
stops speech and dismisses the card. Reveal the new DMG in Finder.
