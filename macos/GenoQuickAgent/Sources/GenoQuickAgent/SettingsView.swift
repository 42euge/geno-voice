import AVFoundation
import SwiftUI

struct SettingsView: View {
    @AppStorage(SpeechPreferences.autoSpeakKey) private var autoSpeak = true
    @AppStorage(SpeechPreferences.voiceIdentifierKey) private var voiceIdentifier = ""
    @AppStorage(SpeechPreferences.rateKey) private var speechRate = SpeechPreferences.defaultRate

    private var englishVoices: [AVSpeechSynthesisVoice] {
        AVSpeechSynthesisVoice.speechVoices()
            .filter { $0.language.lowercased().hasPrefix("en") }
            .sorted {
                $0.name.localizedStandardCompare($1.name) == .orderedAscending
            }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 22) {
            HStack(spacing: 12) {
                Image(systemName: "sparkles")
                    .font(.system(size: 17, weight: .semibold))
                    .foregroundStyle(.white)
                    .frame(width: 38, height: 38)
                    .background(
                        LinearGradient(
                            colors: [Color(red: 0.58, green: 0.46, blue: 1), Color(red: 0.36, green: 0.23, blue: 0.86)],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        ),
                        in: RoundedRectangle(cornerRadius: 11)
                    )
                VStack(alignment: .leading, spacing: 2) {
                    Text(AppVersion.current.applicationName)
                        .font(.system(size: 17, weight: .semibold))
                    Text("One shortcut, one question, one concise answer.")
                        .font(.system(size: 12))
                        .foregroundStyle(.secondary)
                }
            }

            GroupBox("Language model backend") {
                VStack(alignment: .leading, spacing: 13) {
                    LabeledContent("Backend") {
                        Text("Bundled Claude Agent SDK")
                    }
                    LabeledContent("Model") {
                        Text("sonnet via BlueGPT")
                    }
                    LabeledContent("Configuration") {
                        Text("~/.zshrc")
                            .font(.system(.body, design: .monospaced))
                    }
                }
                .padding(10)
            }

            GroupBox("Speech") {
                VStack(alignment: .leading, spacing: 13) {
                    Toggle("Automatically speak answers", isOn: $autoSpeak)

                    LabeledContent("Voice") {
                        Picker("Voice", selection: $voiceIdentifier) {
                            Text("System Default").tag("")
                            ForEach(englishVoices, id: \.identifier) { voice in
                                Text("\(voice.name) (\(voice.language))")
                                    .tag(voice.identifier)
                            }
                        }
                        .labelsHidden()
                        .frame(width: 260)
                    }

                    LabeledContent("Rate") {
                        HStack(spacing: 10) {
                            Slider(value: $speechRate, in: SpeechPreferences.rateRange)
                            Text(speechRate, format: .number.precision(.fractionLength(2)))
                                .monospacedDigit()
                                .frame(width: 34, alignment: .trailing)
                        }
                        .frame(width: 260)
                    }
                }
                .padding(10)
            }

            Text("The app sources your BlueGPT endpoint and credentials from ~/.zshrc for each question. It does not use macOS Keychain and does not require Ollama, Node, or Claude Code to be installed.")
                .font(.system(size: 11))
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            VStack(alignment: .leading, spacing: 7) {
                Label("Option–Space opens the overlay from any app.", systemImage: "command")
                Label("Speech recognition stays on this Mac.", systemImage: "lock.shield")
                Label("Only the transcribed question is sent to your configured LLM endpoint.", systemImage: "network")
            }
            .font(.system(size: 11))
            .foregroundStyle(.secondary)

            Spacer()
        }
        .padding(24)
        .frame(width: 560, height: 580)
    }
}
