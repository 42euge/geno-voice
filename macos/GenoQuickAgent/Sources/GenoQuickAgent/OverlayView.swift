import SwiftUI

struct OverlayView: View {
    @ObservedObject var model: AgentCoordinator

    var body: some View {
        Group {
            switch model.phase {
            case .listening:
                listeningView
            case .transcribing:
                progressView(title: "Finishing your question", detail: model.partialTranscript)
            case .thinking:
                progressView(title: "Finding the short version", detail: model.question)
            case .answer:
                answerView
            case .error:
                errorView
            case .hidden:
                Color.clear
            }
        }
        .padding(8)
        .foregroundStyle(Color(nsColor: OverlayPalette.primaryText))
        .environment(\.colorScheme, .dark)
        .animation(.easeOut(duration: 0.22), value: model.phase)
    }

    private var listeningView: some View {
        HStack(spacing: 13) {
            BrandMark()
            VStack(alignment: .leading, spacing: 2) {
                Text("Listening…")
                    .font(.system(size: 13, weight: .semibold))
                Text(
                    model.partialTranscript.isEmpty
                        ? AppVersion.current.applicationName
                        : model.partialTranscript
                )
                    .font(.system(size: 11))
                    .foregroundStyle(Color(nsColor: OverlayPalette.secondaryText))
                    .lineLimit(1)
            }
            .frame(maxWidth: 150, alignment: .leading)
            AudioBars(level: model.audioLevel)
                .frame(maxWidth: .infinity)
            ShortcutHint(text: "⌥ Space")
        }
        .padding(.horizontal, 14)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .quickAgentSurface(radius: 34)
        .contentShape(Rectangle())
        .onTapGesture { model.stopListening() }
    }

    private func progressView(title: String, detail: String) -> some View {
        HStack(spacing: 13) {
            BrandMark()
            VStack(alignment: .leading, spacing: 3) {
                Text("\(title) · \(AppVersion.current.label)")
                    .font(.system(size: 13, weight: .semibold))
                Text(detail.isEmpty ? "One moment…" : detail)
                    .font(.system(size: 11))
                    .foregroundStyle(Color(nsColor: OverlayPalette.secondaryText))
                    .lineLimit(1)
            }
            Spacer(minLength: 12)
            ThinkingDots()
            ShortcutHint(text: "⌥ Space")
        }
        .padding(.horizontal, 15)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .quickAgentSurface(radius: 38)
    }

    private var answerView: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 11) {
                BrandMark()
                VStack(alignment: .leading, spacing: 1) {
                    Text("Quick answer")
                        .font(.system(size: 13, weight: .semibold))
                    Text(AppVersion.current.applicationName)
                        .font(.system(size: 10))
                        .foregroundStyle(Color(nsColor: OverlayPalette.secondaryText))
                }
                Spacer()
                Button {
                    model.cancelAndDismiss()
                } label: {
                    Image(systemName: "xmark")
                        .font(.system(size: 10, weight: .bold))
                        .frame(width: 25, height: 25)
                }
                .buttonStyle(QuietIconButtonStyle())
            }
            .padding(.horizontal, 17)
            .frame(height: 58)

            Divider().opacity(0.35)

            VStack(alignment: .leading, spacing: 13) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("YOU ASKED")
                        .font(.system(size: 9, weight: .bold))
                        .tracking(1.2)
                        .foregroundStyle(Color(nsColor: OverlayPalette.secondaryText))
                    Text(model.question)
                        .font(.system(size: 12))
                        .foregroundStyle(Color(nsColor: OverlayPalette.secondaryText))
                        .lineLimit(2)
                }

                ScrollView {
                    Text(
                        HighlightedAnswerText.attributed(
                            text: model.answer,
                            range: model.spokenRange
                        )
                    )
                        .font(.system(size: 15))
                        .lineSpacing(5)
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }

                HStack(spacing: 8) {
                    ActionButton(title: "Copy answer", systemImage: "doc.on.doc") {
                        model.copyAnswer()
                    }
                    ActionButton(
                        title: model.isSpeaking ? "Stop" : "Speak",
                        systemImage: model.isSpeaking ? "stop.fill" : "speaker.wave.2"
                    ) {
                        model.toggleAnswerPlayback()
                    }
                    ActionButton(title: "Ask another", systemImage: "mic") {
                        model.startQuestion()
                    }
                    Spacer()
                    Text("esc to close")
                        .font(.system(size: 10))
                        .foregroundStyle(Color(nsColor: OverlayPalette.tertiaryText))
                }
            }
            .padding(17)
        }
        .quickAgentSurface(radius: 22)
    }

    private var errorView: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(spacing: 11) {
                BrandMark(symbol: "exclamationmark")
                VStack(alignment: .leading, spacing: 2) {
                    Text("GenoVoice \(AppVersion.current.label) couldn’t finish")
                        .font(.system(size: 14, weight: .semibold))
                    Text(model.errorMessage)
                        .font(.system(size: 12))
                        .foregroundStyle(Color(nsColor: OverlayPalette.secondaryText))
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            HStack(spacing: 8) {
                if let destination = model.permissionDestination {
                    ActionButton(title: destination.actionTitle, systemImage: "gear") {
                        model.openPermissionSettings()
                    }
                } else {
                    ActionButton(title: "Try again", systemImage: "arrow.clockwise") {
                        model.startQuestion()
                    }
                }
                Spacer()
                Button("Close") { model.cancelAndDismiss() }
                    .buttonStyle(.plain)
                    .font(.system(size: 11))
                    .foregroundStyle(Color(nsColor: OverlayPalette.secondaryText))
            }
        }
        .padding(17)
        .frame(maxWidth: .infinity, alignment: .topLeading)
        .quickAgentSurface(radius: 22)
    }
}

private struct BrandMark: View {
    var symbol = "sparkles"

    var body: some View {
        Image(systemName: symbol)
            .font(.system(size: 13, weight: .semibold))
            .foregroundStyle(.white)
            .frame(width: 29, height: 29)
            .background(
                LinearGradient(
                    colors: [Color(red: 0.58, green: 0.46, blue: 1), Color(red: 0.36, green: 0.23, blue: 0.86)],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                ),
                in: RoundedRectangle(cornerRadius: 9, style: .continuous)
            )
            .shadow(color: Color.purple.opacity(0.28), radius: 10, y: 4)
    }
}

private struct AudioBars: View {
    let level: Float
    private let factors: [CGFloat] = [0.35, 0.7, 0.48, 0.92, 0.58, 0.78, 1, 0.52, 0.86, 0.43, 0.68, 0.38]

    var body: some View {
        HStack(spacing: 3) {
            ForEach(Array(factors.enumerated()), id: \.offset) { index, factor in
                Capsule()
                    .fill(
                        LinearGradient(
                            colors: [.white.opacity(0.88), Color(red: 0.48, green: 0.36, blue: 0.95)],
                            startPoint: .top,
                            endPoint: .bottom
                        )
                    )
                    .frame(width: 3, height: barHeight(factor: factor, index: index))
            }
        }
        .frame(height: 32)
        .animation(.easeOut(duration: 0.09), value: level)
    }

    private func barHeight(factor: CGFloat, index: Int) -> CGFloat {
        let floor = 5 + CGFloat(index % 3)
        return floor + CGFloat(level) * 25 * factor
    }
}

private struct ThinkingDots: View {
    @State private var active = false

    var body: some View {
        HStack(spacing: 5) {
            ForEach(0..<3, id: \.self) { index in
                Circle()
                    .fill(Color(red: 0.59, green: 0.49, blue: 0.95))
                    .frame(width: 6, height: 6)
                    .offset(y: active ? (index == 1 ? -3 : 0) : (index == 1 ? 0 : -2))
                    .opacity(active ? (index == 1 ? 1 : 0.35) : (index == 1 ? 0.35 : 0.8))
            }
        }
        .onAppear {
            withAnimation(.easeInOut(duration: 0.55).repeatForever(autoreverses: true)) {
                active = true
            }
        }
    }
}

private struct ShortcutHint: View {
    let text: String

    var body: some View {
        Text(text)
            .font(.system(size: 9, design: .monospaced))
            .foregroundStyle(Color(nsColor: OverlayPalette.secondaryText))
            .padding(.horizontal, 7)
            .padding(.vertical, 4)
            .background(Color.white.opacity(0.055), in: RoundedRectangle(cornerRadius: 6))
            .overlay(RoundedRectangle(cornerRadius: 6).stroke(Color.white.opacity(0.08)))
    }
}

private struct ActionButton: View {
    let title: String
    let systemImage: String
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Label(title, systemImage: systemImage)
                .font(.system(size: 11, weight: .medium))
                .padding(.horizontal, 10)
                .padding(.vertical, 7)
                .background(Color.white.opacity(0.055), in: RoundedRectangle(cornerRadius: 8))
                .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.white.opacity(0.08)))
        }
        .buttonStyle(.plain)
        .foregroundStyle(Color(nsColor: OverlayPalette.secondaryText))
    }
}

private struct QuietIconButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .foregroundStyle(Color(nsColor: OverlayPalette.secondaryText))
            .background(Color.white.opacity(configuration.isPressed ? 0.1 : 0.04), in: Circle())
    }
}

private extension View {
    func quickAgentSurface(radius: CGFloat) -> some View {
        self
            .background(
                LinearGradient(
                    colors: [
                        Color(nsColor: OverlayPalette.surfaceTop),
                        Color(nsColor: OverlayPalette.surfaceBottom),
                    ],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                ),
                in: RoundedRectangle(cornerRadius: radius, style: .continuous)
            )
            .overlay(
                RoundedRectangle(cornerRadius: radius, style: .continuous)
                    .stroke(Color(nsColor: OverlayPalette.border), lineWidth: 1)
            )
            .shadow(color: .black.opacity(0.48), radius: 28, y: 14)
    }
}
