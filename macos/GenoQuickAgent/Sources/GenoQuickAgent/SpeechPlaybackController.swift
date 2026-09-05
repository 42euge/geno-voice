import AVFoundation
import Foundation

@MainActor
protocol SpeechPlaybackControlling: AnyObject {
    var onSpeakingChanged: ((Bool) -> Void)? { get set }
    var onSpokenRangeChanged: ((NSRange?) -> Void)? { get set }

    func playAutomatically(_ text: String)
    func toggle(_ text: String)
    func stop()
}

@MainActor
final class SpeechPlaybackController: NSObject, SpeechPlaybackControlling {
    var onSpeakingChanged: ((Bool) -> Void)?
    var onSpokenRangeChanged: ((NSRange?) -> Void)?

    private let synthesizer: AVSpeechSynthesizer
    private let defaults: UserDefaults

    init(
        synthesizer: AVSpeechSynthesizer = AVSpeechSynthesizer(),
        defaults: UserDefaults = .standard
    ) {
        self.synthesizer = synthesizer
        self.defaults = defaults
        super.init()
        synthesizer.delegate = self
    }

    func playAutomatically(_ text: String) {
        guard SpeechPreferences.settings(in: defaults).autoSpeak else { return }
        speak(text)
    }

    func toggle(_ text: String) {
        if synthesizer.isSpeaking {
            stop()
        } else {
            speak(text)
        }
    }

    func stop() {
        if synthesizer.isSpeaking || synthesizer.isPaused {
            synthesizer.stopSpeaking(at: .immediate)
        }
        publishStopped()
    }

    private func speak(_ text: String) {
        let cleaned = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !cleaned.isEmpty else { return }

        stop()
        let settings = SpeechPreferences.settings(in: defaults)
        let utterance = AVSpeechUtterance(string: cleaned)
        utterance.rate = Float(settings.rate)
        if !settings.voiceIdentifier.isEmpty,
           let voice = AVSpeechSynthesisVoice(identifier: settings.voiceIdentifier) {
            utterance.voice = voice
        } else {
            utterance.voice = AVSpeechSynthesisVoice(language: "en-US")
        }
        synthesizer.speak(utterance)
    }

    private func publishStopped() {
        onSpeakingChanged?(false)
        onSpokenRangeChanged?(nil)
    }
}

extension SpeechPlaybackController: AVSpeechSynthesizerDelegate {
    nonisolated func speechSynthesizer(
        _ synthesizer: AVSpeechSynthesizer,
        didStart utterance: AVSpeechUtterance
    ) {
        Task { @MainActor [weak self] in
            self?.onSpeakingChanged?(true)
        }
    }

    nonisolated func speechSynthesizer(
        _ synthesizer: AVSpeechSynthesizer,
        willSpeakRangeOfSpeechString characterRange: NSRange,
        utterance: AVSpeechUtterance
    ) {
        Task { @MainActor [weak self] in
            self?.onSpokenRangeChanged?(characterRange)
        }
    }

    nonisolated func speechSynthesizer(
        _ synthesizer: AVSpeechSynthesizer,
        didFinish utterance: AVSpeechUtterance
    ) {
        Task { @MainActor [weak self] in
            self?.publishStopped()
        }
    }

    nonisolated func speechSynthesizer(
        _ synthesizer: AVSpeechSynthesizer,
        didCancel utterance: AVSpeechUtterance
    ) {
        Task { @MainActor [weak self] in
            self?.publishStopped()
        }
    }
}
