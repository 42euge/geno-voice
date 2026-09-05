import AppKit
import Foundation
import SwiftUI

private var failures = 0

private func expect(_ condition: @autoclosure () -> Bool, _ message: String) {
    if !condition() {
        failures += 1
        fputs("FAIL: \(message)\n", stderr)
    }
}

private func expectEqual(_ actual: CGFloat, _ expected: CGFloat, _ message: String) {
    expect(abs(actual - expected) < 0.001, "\(message) — expected \(expected), got \(actual)")
}

private final class AsyncResultBox<Value>: @unchecked Sendable {
    private let lock = NSLock()
    private var stored: Result<Value, Error>?

    func set(_ result: Result<Value, Error>) {
        lock.lock()
        stored = result
        lock.unlock()
    }

    func get() -> Result<Value, Error>? {
        lock.lock()
        defer { lock.unlock() }
        return stored
    }
}

private func waitForResult<Value>(
    timeout: TimeInterval = 5,
    operation: @escaping @Sendable () async throws -> Value
) -> Result<Value, Error>? {
    let box = AsyncResultBox<Value>()
    let semaphore = DispatchSemaphore(value: 0)
    Task.detached {
        do {
            box.set(.success(try await operation()))
        } catch {
            box.set(.failure(error))
        }
        semaphore.signal()
    }
    guard semaphore.wait(timeout: .now() + timeout) == .success else { return nil }
    return box.get()
}

@MainActor
private final class FakeSpeechPlayback: SpeechPlaybackControlling {
    var onSpeakingChanged: ((Bool) -> Void)?
    var onSpokenRangeChanged: ((NSRange?) -> Void)?
    private(set) var automaticTexts: [String] = []
    private(set) var toggledTexts: [String] = []
    private(set) var stopCount = 0

    func playAutomatically(_ text: String) {
        automaticTexts.append(text)
    }

    func toggle(_ text: String) {
        toggledTexts.append(text)
    }

    func stop() {
        stopCount += 1
    }
}

@MainActor
private final class FakeSpeechRecorder: SpeechRecording {
    var onLevel: ((Float) -> Void)?
    var onPartialTranscript: ((String) -> Void)?
    var onCaptureEnded: (() -> Void)?
    var onComplete: ((Result<String, Error>) -> Void)?

    func start() {}
    func stop() {}
    func cancel() {}
}

// Release identity is user-visible everywhere, including Finder. A stale app
// must be obvious from a screenshot or from the mounted installer alone.
let releaseVersion = AppVersion(shortVersion: "0.2.0", buildNumber: "9")
expect(releaseVersion.label == "v0.2.0 (9)", "version label should include semantic and build versions")
expect(
    releaseVersion.applicationName == "GenoVoice v0.2.0 (9)",
    "application name should include the complete version label"
)

// Native speech defaults must be useful without setup while still respecting
// the safe rate range expected by AVSpeechSynthesizer.
let speechDefaultsSuite = "GenoQuickAgentSpeechPreferences-\(UUID().uuidString)"
let speechDefaults = UserDefaults(suiteName: speechDefaultsSuite)!
defer { speechDefaults.removePersistentDomain(forName: speechDefaultsSuite) }
SpeechPreferences.registerDefaults(in: speechDefaults)

var speechSettings = SpeechPreferences.settings(in: speechDefaults)
expect(speechSettings.autoSpeak, "auto-speak should be enabled by default")
expect(speechSettings.voiceIdentifier.isEmpty, "the default voice should follow the system")
expect(abs(speechSettings.rate - 0.48) < 0.0001, "the default speaking rate should be 0.48")

speechDefaults.set(0.10, forKey: SpeechPreferences.rateKey)
speechSettings = SpeechPreferences.settings(in: speechDefaults)
expect(abs(speechSettings.rate - 0.35) < 0.0001, "speaking rate should clamp to 0.35")

speechDefaults.set(0.90, forKey: SpeechPreferences.rateKey)
speechSettings = SpeechPreferences.settings(in: speechDefaults)
expect(abs(speechSettings.rate - 0.60) < 0.0001, "speaking rate should clamp to 0.60")

// AVSpeechSynthesizer reports UTF-16 offsets. Emoji before the active word must
// not shift the Swift String range or crash the answer renderer.
let emojiAnswer = "Hi 👋 there"
let thereRange = (emojiAnswer as NSString).range(of: "there")
expect(
    SpokenWordRange.substring(thereRange, in: emojiAnswer) == "there",
    "UTF-16 spoken range should survive an emoji prefix"
)
expect(
    SpokenWordRange.substring(NSRange(location: 99, length: 4), in: emojiAnswer) == nil,
    "out-of-bounds spoken range should be ignored"
)
expect(
    HighlightedAnswerText.activeWord(text: emojiAnswer, range: thereRange) == "there",
    "highlighted answer should expose the word currently being spoken"
)
expect(
    HighlightedAnswerText.activeWord(
        text: emojiAnswer,
        range: NSRange(location: 99, length: 4)
    ) == nil,
    "highlighted answer should ignore stale speech ranges"
)

for (name, color) in [
    ("top", OverlayPalette.surfaceTop),
    ("bottom", OverlayPalette.surfaceBottom),
] {
    guard let rgb = color.usingColorSpace(.sRGB) else {
        expect(false, "\(name) surface color should convert to sRGB")
        continue
    }
    expect(rgb.redComponent < 0.16, "\(name) surface red should stay near-black")
    expect(rgb.greenComponent < 0.16, "\(name) surface green should stay near-black")
    expect(rgb.blueComponent < 0.16, "\(name) surface blue should stay near-black")
    expect(rgb.alphaComponent >= 0.96, "\(name) surface should stay opaque")
}

// Playback is a side effect behind a seam. Coordinator state remains the
// consumer-visible behavior while the fake replaces only macOS audio output.
MainActor.assumeIsolated {
    let recorder = FakeSpeechRecorder()
    let playback = FakeSpeechPlayback()
    let coordinator = AgentCoordinator(speechRecorder: recorder, speechPlayback: playback)
    coordinator.receiveAnswer("A concise spoken answer.")
    expect(coordinator.phase == .answer, "receiving an answer should show the answer phase")
    expect(coordinator.answer == "A concise spoken answer.", "coordinator should publish the answer")
    expect(
        playback.automaticTexts == ["A concise spoken answer."],
        "receiving an answer should request automatic playback"
    )

    let activeRange = NSRange(location: 2, length: 7)
    playback.onSpeakingChanged?(true)
    playback.onSpokenRangeChanged?(activeRange)
    expect(coordinator.isSpeaking, "playback callback should publish speaking state")
    expect(coordinator.spokenRange == activeRange, "playback callback should publish active word range")

    coordinator.toggleAnswerPlayback()
    expect(
        playback.toggledTexts == ["A concise spoken answer."],
        "answer playback control should toggle the current answer"
    )

    coordinator.cancelAndDismiss()
    expect(playback.stopCount == 1, "dismiss should stop speech")
    expect(!coordinator.isSpeaking, "dismiss should clear speaking state")
    expect(coordinator.spokenRange == nil, "dismiss should clear the active word")

    coordinator.receiveAnswer("Another answer.")
    playback.onSpeakingChanged?(true)
    playback.onSpokenRangeChanged?(NSRange(location: 0, length: 7))
    coordinator.presentError("Replacement error")
    expect(playback.stopCount == 2, "an error should stop speech")
    expect(!coordinator.isSpeaking, "an error should clear speaking state")
    expect(coordinator.spokenRange == nil, "an error should clear the active word")

    coordinator.receiveAnswer("Answer before a new question.")
    playback.onSpeakingChanged?(true)
    playback.onSpokenRangeChanged?(NSRange(location: 0, length: 6))
    coordinator.startQuestion()
    expect(playback.stopCount == 3, "a new question should stop speech")
    expect(!coordinator.isSpeaking, "a new question should clear speaking state")
    expect(coordinator.spokenRange == nil, "a new question should clear the active word")
    coordinator.cancelAndDismiss()
}

// Regression: the Swift client must source the configured zshrc and exchange
// the real JSON-lines protocol with a separate backend process.
let backendFixtureDirectory = FileManager.default.temporaryDirectory
    .appendingPathComponent("GenoQuickAgentBackend-\(UUID().uuidString)", isDirectory: true)
try FileManager.default.createDirectory(
    at: backendFixtureDirectory,
    withIntermediateDirectories: true
)
defer { try? FileManager.default.removeItem(at: backendFixtureDirectory) }

let fixtureZshrc = backendFixtureDirectory.appendingPathComponent("zshrc")
try """
export GENO_TEST_CONFIG_LOADED=1
unset ANTHROPIC_AUTH_TOKEN ANTHROPIC_API_KEY ANTHROPIC_BASE_URL
export OPENAI_API_KEY=provider-neutral-test-token
export OPENAI_BASE_URL=https://openai-compatible.example/v1
""".write(
    to: fixtureZshrc,
    atomically: true,
    encoding: .utf8
)
let fixtureBackend = backendFixtureDirectory.appendingPathComponent("fake-backend.sh")
try """
#!/bin/sh
payload="$(cat)"
request_id="$(printf '%s' "$payload" | sed -E 's/.*"id":"([^"]+)".*/\\1/')"
if [ "$GENO_TEST_CONFIG_LOADED" != "1" ]; then
  printf '{"id":"%s","error":{"code":"missing_config","message":"zshrc was not sourced"}}\\n' "$request_id"
  exit 1
fi
if [ -n "${ANTHROPIC_AUTH_TOKEN:-}" ] || [ -n "${ANTHROPIC_BASE_URL:-}" ]; then
  printf '{"id":"%s","error":{"code":"cross_mapped_config","message":"backend remapped configuration for another API"}}\\n' "$request_id"
  exit 1
fi
case "$payload" in
  *'"question":"What is two plus two?"'*)
    printf '{"id":"%s","answer":"four"}\\n' "$request_id"
    ;;
  *)
    printf '{"id":"%s","error":{"code":"wrong_question","message":"question did not cross protocol"}}\\n' "$request_id"
    exit 1
    ;;
esac
""".write(to: fixtureBackend, atomically: true, encoding: .utf8)

let fixtureClient = ClaudeBackendClient(
    configuration: ClaudeBackendConfiguration(
        shellURL: URL(fileURLWithPath: "/bin/zsh"),
        runtimeURL: URL(fileURLWithPath: "/bin/sh"),
        scriptURL: fixtureBackend,
        zshrcURL: fixtureZshrc,
        model: "sonnet",
        timeout: 2
    )
)
let fixtureResult = waitForResult {
    try await fixtureClient.ask("What is two plus two?")
}
switch fixtureResult {
case .success(let answer):
    expect(answer == "four", "Claude backend client should return the subprocess answer")
case .failure(let error):
    expect(false, "Claude backend client failed: \(error.localizedDescription)")
case nil:
    expect(false, "Claude backend client timed out")
}

// Regression: restarting voice capture must never reuse an AVAudioEngine that
// may still own a recording tap. Each capture generation gets a fresh engine.
var createdEngineID = 0
let engineStore = CaptureEngineStore<Int> {
    createdEngineID += 1
    return createdEngineID
}
let firstEngine = engineStore.begin()
engineStore.end()
let secondEngine = engineStore.begin()
expect(firstEngine == 1, "first capture should use the first engine generation")
expect(secondEngine == 2, "second capture should use a fresh engine generation")
expect(firstEngine != secondEngine, "capture restarts must not reuse an engine")

// Regression: SwiftUI hosting previously expanded the error panel to the full
// 1,400-point screen height after a drag. The panel is fixed-size and cannot be
// moved; its hosting controller must not resize the window from intrinsic size.
let compactErrorSize = OverlayLayout.preferredSize(for: .error)
expectEqual(compactErrorSize.width, 460, "compact error width")
expectEqual(compactErrorSize.height, 138, "compact error height")

MainActor.assumeIsolated {
    let panel = NSPanel(
        contentRect: NSRect(origin: .zero, size: compactErrorSize),
        styleMask: [.borderless, .nonactivatingPanel],
        backing: .buffered,
        defer: false
    )
    let hostingController = NSHostingController(rootView: EmptyView())
    OverlayWindowPolicy.lock(
        panel: panel,
        hostingController: hostingController,
        to: compactErrorSize
    )

    expect(!panel.isMovable, "overlay panel should not be movable")
    expect(!panel.isMovableByWindowBackground, "overlay background should not drag the panel")
    expectEqual(panel.minSize.width, 460, "panel minimum width")
    expectEqual(panel.minSize.height, 138, "panel minimum height")
    expectEqual(panel.maxSize.width, 460, "panel maximum width")
    expectEqual(panel.maxSize.height, 138, "panel maximum height")
    expect(hostingController.sizingOptions.isEmpty, "hosting controller should not autosize the panel")
}

// Regression: a preferred answer card larger than the usable screen must be
// reduced and inset on every edge instead of extending off-screen.
let constrainedFrame = OverlayLayout.frame(
    preferredSize: NSSize(width: 620, height: 360),
    within: NSRect(x: 0, y: 0, width: 500, height: 300)
)
expectEqual(constrainedFrame.origin.x, 16, "constrained frame x origin")
expectEqual(constrainedFrame.origin.y, 16, "constrained frame y origin")
expectEqual(constrainedFrame.width, 468, "constrained frame width")
expectEqual(constrainedFrame.height, 268, "constrained frame height")

// Regression: the normal-sized overlay stays centered and keeps its preferred
// bottom offset when the screen has enough room.
let roomyFrame = OverlayLayout.frame(
    preferredSize: NSSize(width: 620, height: 360),
    within: NSRect(x: 1440, y: 24, width: 1440, height: 876)
)
expectEqual(roomyFrame.origin.x, 1850, "roomy frame x origin")
expectEqual(roomyFrame.origin.y, 102, "roomy frame y origin")
expectEqual(roomyFrame.width, 620, "roomy frame width")
expectEqual(roomyFrame.height, 360, "roomy frame height")

// Regression: Option–Space starts only while hidden. Every visible phase uses
// the same dismiss behavior, including the permission error and answer cards.
expect(ShortcutPolicy.action(for: .hidden) == .start, "hidden shortcut action should start")
for phase in [AgentPhase.listening, .transcribing, .thinking, .answer, .error] {
    expect(ShortcutPolicy.action(for: phase) == .dismiss, "\(phase) shortcut action should dismiss")
}

MainActor.assumeIsolated {
    let coordinator = AgentCoordinator()
    coordinator.presentError(SpeechRecorderError.microphonePermissionDenied)
    expect(
        coordinator.permissionDestination == .microphone,
        "coordinator should preserve the permission destination for the error card"
    )
    coordinator.toggleFromShortcut()
    expect(coordinator.phase == .hidden, "shortcut should dismiss a visible coordinator error")
}

// Regression: macOS can authorize Speech Recognition while system Dictation
// remains disabled. The local recognizer then reports kLSRErrorDomain/201;
// turn that opaque framework error into direct, actionable settings guidance.
MainActor.assumeIsolated {
    let coordinator = AgentCoordinator()
    let frameworkError = NSError(
        domain: "kLSRErrorDomain",
        code: 201,
        userInfo: [NSLocalizedDescriptionKey: "Siri and Dictation are disabled"]
    )
    coordinator.presentError(frameworkError)
    expect(
        coordinator.errorMessage ==
            "Dictation is off. Open Keyboard Settings and turn on Dictation, then try again.",
        "disabled Dictation should produce actionable guidance"
    )
    expect(
        coordinator.permissionDestination?.settingsURL.absoluteString ==
            "x-apple.systempreferences:com.apple.Keyboard-Settings.extension",
        "disabled Dictation should deep-link to Keyboard settings"
    )
    expect(
        coordinator.permissionDestination?.actionTitle == "Open Keyboard Settings",
        "disabled Dictation should label the settings action clearly"
    )
}

// Regression: permission errors retain a typed destination so the UI can open
// the exact Privacy & Security pane instead of showing dead-end prose.
let microphoneDestination = SpeechRecorderError.microphonePermissionDenied.permissionDestination
expect(microphoneDestination == .microphone, "microphone denial should retain microphone destination")
expect(
    microphoneDestination?.settingsURL.absoluteString ==
        "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone",
    "microphone destination should deep-link to Microphone settings"
)

let speechDestination = SpeechRecorderError.speechPermissionDenied.permissionDestination
expect(speechDestination == .speechRecognition, "speech denial should retain speech-recognition destination")
expect(
    speechDestination?.settingsURL.absoluteString ==
        "x-apple.systempreferences:com.apple.preference.security?Privacy_SpeechRecognition",
    "speech destination should deep-link to Speech Recognition settings"
)
expect(
    SpeechRecorderError.recognizerUnavailable.permissionDestination == nil,
    "non-permission speech failures should not offer a settings destination"
)

if failures > 0 {
    fputs("\(failures) regression test(s) failed.\n", stderr)
    exit(1)
}

print("All GenoVoice regression tests passed.")
