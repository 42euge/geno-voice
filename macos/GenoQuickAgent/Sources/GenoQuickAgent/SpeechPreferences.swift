import Foundation

struct SpeechPlaybackSettings: Equatable, Sendable {
    let autoSpeak: Bool
    let voiceIdentifier: String
    let rate: Double
}

enum SpeechPreferences {
    static let autoSpeakKey = "speech.autoSpeak"
    static let voiceIdentifierKey = "speech.voiceIdentifier"
    static let rateKey = "speech.rate"
    static let defaultRate = 0.48
    static let rateRange = 0.35...0.60

    static func registerDefaults(in defaults: UserDefaults = .standard) {
        defaults.register(defaults: [
            autoSpeakKey: true,
            voiceIdentifierKey: "",
            rateKey: defaultRate,
        ])
    }

    static func settings(in defaults: UserDefaults = .standard) -> SpeechPlaybackSettings {
        let storedRate = defaults.double(forKey: rateKey)
        return SpeechPlaybackSettings(
            autoSpeak: defaults.bool(forKey: autoSpeakKey),
            voiceIdentifier: defaults.string(forKey: voiceIdentifierKey) ?? "",
            rate: min(max(storedRate, rateRange.lowerBound), rateRange.upperBound)
        )
    }
}
