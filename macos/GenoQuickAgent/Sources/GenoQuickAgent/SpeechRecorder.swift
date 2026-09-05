import AVFoundation
import Foundation
import Speech

@MainActor
protocol SpeechRecording: AnyObject {
    var onLevel: ((Float) -> Void)? { get set }
    var onPartialTranscript: ((String) -> Void)? { get set }
    var onCaptureEnded: (() -> Void)? { get set }
    var onComplete: ((Result<String, Error>) -> Void)? { get set }

    func start()
    func stop()
    func cancel()
}

enum SpeechRecorderError: LocalizedError {
    case microphoneUnavailable
    case microphonePermissionDenied
    case dictationDisabled
    case recognizerUnavailable
    case onDeviceRecognitionUnavailable
    case speechPermissionDenied
    case startFailed(String)

    var errorDescription: String? {
        switch self {
        case .microphoneUnavailable:
            return "No microphone input is available."
        case .microphonePermissionDenied:
            return "Microphone access is off. Enable GenoVoice in System Settings → Privacy & Security → Microphone."
        case .dictationDisabled:
            return "Dictation is off. Open Keyboard Settings and turn on Dictation, then try again."
        case .recognizerUnavailable:
            return "Speech recognition is temporarily unavailable."
        case .onDeviceRecognitionUnavailable:
            return "On-device speech recognition is unavailable for this Mac or language."
        case .speechPermissionDenied:
            return "Speech recognition access is off. Enable GenoVoice in System Settings → Privacy & Security → Speech Recognition."
        case .startFailed(let reason):
            return "Couldn’t start listening: \(reason)"
        }
    }

    static func userFacing(_ error: Error) -> Error {
        let frameworkError = error as NSError
        if frameworkError.domain == "kLSRErrorDomain", frameworkError.code == 201 {
            return SpeechRecorderError.dictationDisabled
        }
        return error
    }
}

@MainActor
final class SpeechRecorder: NSObject {
    var onLevel: ((Float) -> Void)?
    var onPartialTranscript: ((String) -> Void)?
    var onCaptureEnded: (() -> Void)?
    var onComplete: ((Result<String, Error>) -> Void)?

    private let audioEngines = CaptureEngineStore(makeEngine: AVAudioEngine.init)
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?
    private var meterTimer: Timer?
    private var finalizeTimer: Timer?
    private var latestTranscript = ""
    private var heardVoice = false
    private var lastVoiceAt = Date()
    private var startedAt = Date()
    private var isCapturing = false
    private var isAwaitingFinalResult = false
    private var didComplete = false
    private var tapInstalled = false
    private var generation = 0

    func start() {
        cancel()
        generation += 1
        let activeGeneration = generation
        didComplete = false

        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized:
            requestSpeechAuthorization(generation: activeGeneration)
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .audio) { [weak self] granted in
                Task { @MainActor in
                    guard self?.generation == activeGeneration else { return }
                    if granted {
                        self?.requestSpeechAuthorization(generation: activeGeneration)
                    } else {
                        self?.finish(.failure(SpeechRecorderError.microphonePermissionDenied))
                    }
                }
            }
        default:
            finish(.failure(SpeechRecorderError.microphonePermissionDenied))
        }
    }

    func stop() {
        guard isCapturing else { return }
        stopCaptureAndAwaitFinalResult()
    }

    func cancel() {
        generation += 1
        meterTimer?.invalidate()
        meterTimer = nil
        finalizeTimer?.invalidate()
        finalizeTimer = nil
        if let audioEngine = audioEngines.current {
            if audioEngine.isRunning {
                audioEngine.stop()
            }
            removeInputTap(from: audioEngine)
        }
        audioEngines.end()
        recognitionRequest?.endAudio()
        recognitionTask?.cancel()
        recognitionTask = nil
        recognitionRequest = nil
        isCapturing = false
        isAwaitingFinalResult = false
        onLevel?(0)
    }

    private func requestSpeechAuthorization(generation activeGeneration: Int) {
        guard generation == activeGeneration else { return }
        switch SFSpeechRecognizer.authorizationStatus() {
        case .authorized:
            beginRecognition(generation: activeGeneration)
        case .notDetermined:
            SFSpeechRecognizer.requestAuthorization { [weak self] status in
                Task { @MainActor in
                    guard self?.generation == activeGeneration else { return }
                    if status == .authorized {
                        self?.beginRecognition(generation: activeGeneration)
                    } else {
                        self?.finish(.failure(SpeechRecorderError.speechPermissionDenied))
                    }
                }
            }
        default:
            finish(.failure(SpeechRecorderError.speechPermissionDenied))
        }
    }

    private func beginRecognition(generation activeGeneration: Int) {
        guard generation == activeGeneration else { return }
        guard let recognizer = SFSpeechRecognizer(locale: Locale(identifier: "en-US")), recognizer.isAvailable else {
            finish(.failure(SpeechRecorderError.recognizerUnavailable))
            return
        }
        guard recognizer.supportsOnDeviceRecognition else {
            finish(.failure(SpeechRecorderError.onDeviceRecognitionUnavailable))
            return
        }

        let request = SFSpeechAudioBufferRecognitionRequest()
        request.shouldReportPartialResults = true
        request.requiresOnDeviceRecognition = true
        request.taskHint = .dictation
        recognitionRequest = request

        latestTranscript = ""
        heardVoice = false
        lastVoiceAt = Date()
        startedAt = Date()
        isAwaitingFinalResult = false

        let audioEngine = audioEngines.begin()
        let inputNode = audioEngine.inputNode
        let format = inputNode.outputFormat(forBus: 0)
        guard format.sampleRate > 0, format.channelCount > 0 else {
            finish(.failure(SpeechRecorderError.microphoneUnavailable))
            return
        }

        inputNode.installTap(onBus: 0, bufferSize: 1024, format: format) { [weak self, weak request] buffer, _ in
            request?.append(buffer)
            let level = Self.normalizedLevel(in: buffer)
            Task { @MainActor in
                self?.receiveAudioLevel(level)
            }
        }
        tapInstalled = true

        recognitionTask = recognizer.recognitionTask(with: request) { [weak self] result, error in
            Task { @MainActor in
                guard self?.generation == activeGeneration else { return }
                self?.receiveRecognition(result: result, error: error)
            }
        }

        do {
            audioEngine.prepare()
            try audioEngine.start()
            isCapturing = true
            meterTimer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
                Task { @MainActor in
                    self?.checkForEndOfQuestion()
                }
            }
        } catch {
            finish(.failure(SpeechRecorderError.startFailed(error.localizedDescription)))
        }
    }

    private func receiveAudioLevel(_ level: Float) {
        guard isCapturing else { return }
        onLevel?(level)
        if level > 0.16 {
            heardVoice = true
            lastVoiceAt = Date()
        }
    }

    private func receiveRecognition(result: SFSpeechRecognitionResult?, error: Error?) {
        if let result {
            latestTranscript = result.bestTranscription.formattedString
            if !latestTranscript.isEmpty {
                heardVoice = true
                lastVoiceAt = Date()
            }
            onPartialTranscript?(latestTranscript)
            if result.isFinal {
                finish(.success(latestTranscript))
                return
            }
        }

        if let error, !didComplete {
            if isAwaitingFinalResult, !latestTranscript.isEmpty {
                finish(.success(latestTranscript))
            } else if !isAwaitingFinalResult {
                finish(.failure(error))
            }
        }
    }

    private func checkForEndOfQuestion() {
        guard isCapturing else { return }
        let now = Date()
        if heardVoice, now.timeIntervalSince(lastVoiceAt) > 0.95 {
            stopCaptureAndAwaitFinalResult()
        } else if now.timeIntervalSince(startedAt) > 30 {
            stopCaptureAndAwaitFinalResult()
        }
    }

    private func stopCaptureAndAwaitFinalResult() {
        guard isCapturing else { return }
        isCapturing = false
        isAwaitingFinalResult = true
        meterTimer?.invalidate()
        meterTimer = nil
        if let audioEngine = audioEngines.current {
            audioEngine.stop()
            removeInputTap(from: audioEngine)
        }
        recognitionRequest?.endAudio()
        onLevel?(0)
        onCaptureEnded?()

        finalizeTimer = Timer.scheduledTimer(withTimeInterval: 1.5, repeats: false) { [weak self] _ in
            Task { @MainActor in
                guard let self, !self.didComplete else { return }
                self.finish(.success(self.latestTranscript))
            }
        }
    }

    private func finish(_ result: Result<String, Error>) {
        guard !didComplete else { return }
        didComplete = true
        meterTimer?.invalidate()
        meterTimer = nil
        finalizeTimer?.invalidate()
        finalizeTimer = nil
        if let audioEngine = audioEngines.current {
            if audioEngine.isRunning {
                audioEngine.stop()
            }
            removeInputTap(from: audioEngine)
        }
        audioEngines.end()
        recognitionRequest?.endAudio()
        recognitionTask?.cancel()
        recognitionTask = nil
        recognitionRequest = nil
        isCapturing = false
        isAwaitingFinalResult = false
        onLevel?(0)
        onComplete?(result)
    }

    private func removeInputTap(from audioEngine: AVAudioEngine) {
        guard tapInstalled else { return }
        audioEngine.inputNode.removeTap(onBus: 0)
        tapInstalled = false
    }

    private nonisolated static func normalizedLevel(in buffer: AVAudioPCMBuffer) -> Float {
        guard let channel = buffer.floatChannelData?.pointee else { return 0 }
        let count = Int(buffer.frameLength)
        guard count > 0 else { return 0 }
        var sum: Float = 0
        for index in 0..<count {
            let sample = channel[index]
            sum += sample * sample
        }
        let rms = sqrt(sum / Float(count))
        return min(max((rms - 0.008) * 18, 0), 1)
    }
}

extension SpeechRecorder: SpeechRecording {}
