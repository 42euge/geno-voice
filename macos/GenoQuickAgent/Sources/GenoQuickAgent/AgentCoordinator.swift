import AppKit
import Foundation
import OSLog

private let quickAgentLogger = Logger(
    subsystem: "com.geno.quickagent",
    category: "runtime"
)

@MainActor
final class AgentCoordinator: ObservableObject {
    @Published private(set) var phase: AgentPhase = .hidden
    @Published private(set) var partialTranscript = ""
    @Published private(set) var question = ""
    @Published private(set) var answer = ""
    @Published private(set) var errorMessage = ""
    @Published private(set) var permissionDestination: PermissionDestination?
    @Published private(set) var audioLevel: Float = 0
    @Published private(set) var isSpeaking = false
    @Published private(set) var spokenRange: NSRange?

    var onPresent: (() -> Void)?
    var onDismiss: (() -> Void)?

    private let speechRecorder: SpeechRecording
    private let claudeBackend: ClaudeBackendClient
    private let speechPlayback: SpeechPlaybackControlling
    private var requestID = UUID()
    private var answerTask: Task<Void, Never>?

    init(
        speechRecorder: SpeechRecording? = nil,
        claudeBackend: ClaudeBackendClient = ClaudeBackendClient(),
        speechPlayback: SpeechPlaybackControlling? = nil
    ) {
        let recorder = speechRecorder ?? SpeechRecorder()
        let playback = speechPlayback ?? SpeechPlaybackController()
        self.speechRecorder = recorder
        self.claudeBackend = claudeBackend
        self.speechPlayback = playback

        recorder.onLevel = { [weak self] level in
            self?.audioLevel = level
        }
        recorder.onPartialTranscript = { [weak self] transcript in
            self?.partialTranscript = transcript
        }
        recorder.onCaptureEnded = { [weak self] in
            self?.phase = .transcribing
        }
        recorder.onComplete = { [weak self] result in
            self?.handleTranscription(result)
        }
        playback.onSpeakingChanged = { [weak self] isSpeaking in
            self?.isSpeaking = isSpeaking
        }
        playback.onSpokenRangeChanged = { [weak self] range in
            self?.spokenRange = range
        }
    }

    func toggleFromShortcut() {
        switch ShortcutPolicy.action(for: phase) {
        case .start:
            startQuestion()
        case .dismiss:
            cancelAndDismiss()
        }
    }

    func startQuestion() {
        stopAnswerPlayback()
        requestID = UUID()
        answerTask?.cancel()
        answerTask = nil
        speechRecorder.cancel()
        partialTranscript = ""
        question = ""
        answer = ""
        errorMessage = ""
        permissionDestination = nil
        audioLevel = 0
        phase = .listening
        onPresent?()
        speechRecorder.start()
    }

    func stopListening() {
        guard phase == .listening else { return }
        speechRecorder.stop()
    }

    func cancelAndDismiss() {
        stopAnswerPlayback()
        requestID = UUID()
        answerTask?.cancel()
        answerTask = nil
        speechRecorder.cancel()
        phase = .hidden
        permissionDestination = nil
        onDismiss?()
    }

    func copyAnswer() {
        guard !answer.isEmpty else { return }
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(answer, forType: .string)
    }

    func toggleAnswerPlayback() {
        guard !answer.isEmpty else { return }
        speechPlayback.toggle(answer)
    }

    func receiveAnswer(_ response: String) {
        answer = response
        phase = .answer
        speechPlayback.playAutomatically(response)
    }

    func presentError(
        _ message: String,
        permissionDestination: PermissionDestination? = nil
    ) {
        stopAnswerPlayback()
        answerTask?.cancel()
        answerTask = nil
        speechRecorder.cancel()
        errorMessage = message
        self.permissionDestination = permissionDestination
        phase = .error
        onPresent?()
    }

    func presentError(_ error: Error) {
        let diagnostic = error as NSError
        quickAgentLogger.error(
            "GenoVoice error: domain=\(diagnostic.domain, privacy: .public) code=\(diagnostic.code, privacy: .public) description=\(diagnostic.localizedDescription, privacy: .public)"
        )
        let userFacingError = SpeechRecorderError.userFacing(error)
        presentError(
            userFacingError.localizedDescription,
            permissionDestination: (userFacingError as? SpeechRecorderError)?.permissionDestination
        )
    }

    func openPermissionSettings() {
        guard let permissionDestination else { return }
        NSWorkspace.shared.open(permissionDestination.settingsURL)
        cancelAndDismiss()
    }

    private func handleTranscription(_ result: Result<String, Error>) {
        switch result {
        case .failure(let error):
            presentError(error)
        case .success(let transcript):
            let cleaned = transcript.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !cleaned.isEmpty else {
                presentError("I didn’t catch a question. Try again and speak a little closer to the microphone.")
                return
            }
            question = cleaned
            partialTranscript = cleaned
            askLLM(cleaned)
        }
    }

    private func askLLM(_ prompt: String) {
        let activeRequest = UUID()
        requestID = activeRequest
        phase = .thinking

        answerTask?.cancel()
        answerTask = Task {
            do {
                let response = try await claudeBackend.ask(prompt)
                try Task.checkCancellation()
                guard requestID == activeRequest else { return }
                receiveAnswer(response)
            } catch is CancellationError {
                return
            } catch {
                guard requestID == activeRequest else { return }
                presentError(error)
            }
        }
    }

    private func stopAnswerPlayback() {
        speechPlayback.stop()
        isSpeaking = false
        spokenRange = nil
    }
}
