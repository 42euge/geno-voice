enum ShortcutAction: Equatable {
    case start
    case dismiss
}

enum ShortcutPolicy {
    static func action(for phase: AgentPhase) -> ShortcutAction {
        phase == .hidden ? .start : .dismiss
    }
}
