import Foundation

enum PermissionDestination: Equatable {
    case microphone
    case speechRecognition
    case dictation

    var settingsURL: URL {
        switch self {
        case .microphone:
            return URL(
                string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone"
            )!
        case .speechRecognition:
            return URL(
                string: "x-apple.systempreferences:com.apple.preference.security?Privacy_SpeechRecognition"
            )!
        case .dictation:
            return URL(
                string: "x-apple.systempreferences:com.apple.Keyboard-Settings.extension"
            )!
        }
    }

    var actionTitle: String {
        switch self {
        case .microphone:
            return "Open Microphone Settings"
        case .speechRecognition:
            return "Open Speech Recognition Settings"
        case .dictation:
            return "Open Keyboard Settings"
        }
    }
}

extension SpeechRecorderError {
    var permissionDestination: PermissionDestination? {
        switch self {
        case .microphonePermissionDenied:
            return .microphone
        case .speechPermissionDenied:
            return .speechRecognition
        case .dictationDisabled:
            return .dictation
        case .microphoneUnavailable, .recognizerUnavailable,
             .onDeviceRecognitionUnavailable, .startFailed:
            return nil
        }
    }
}
