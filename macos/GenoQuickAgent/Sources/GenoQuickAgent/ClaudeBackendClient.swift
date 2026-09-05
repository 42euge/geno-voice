import Foundation

struct ClaudeBackendConfiguration: Sendable {
    let shellURL: URL
    let runtimeURL: URL
    let scriptURL: URL
    let zshrcURL: URL
    let model: String
    let timeout: TimeInterval

    static var bundled: ClaudeBackendConfiguration {
        let resources = Bundle.main.resourceURL ?? Bundle.main.bundleURL
        let backend = resources.appendingPathComponent("ClaudeBackend", isDirectory: true)
        return ClaudeBackendConfiguration(
            shellURL: URL(fileURLWithPath: "/bin/zsh"),
            runtimeURL: backend.appendingPathComponent("node"),
            scriptURL: backend.appendingPathComponent("claude-backend.mjs"),
            zshrcURL: FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".zshrc"),
            model: "sonnet",
            timeout: 60
        )
    }
}

enum ClaudeBackendError: LocalizedError {
    case missingRuntime
    case missingScript
    case missingZshrc
    case launchFailed(String)
    case timeout
    case malformedResponse
    case backend(code: String, message: String)
    case emptyAnswer

    var errorDescription: String? {
        switch self {
        case .missingRuntime:
            return "The bundled Claude runtime is missing. Reinstall GenoVoice."
        case .missingScript:
            return "The bundled Claude backend is missing. Reinstall GenoVoice."
        case .missingZshrc:
            return "GenoVoice couldn’t find ~/.zshrc with your Claude endpoint configuration."
        case .launchFailed(let reason):
            return "Couldn’t start the Claude backend: \(reason)"
        case .timeout:
            return "Claude took longer than one minute to answer. Try again."
        case .malformedResponse:
            return "The Claude backend returned a response GenoVoice couldn’t read."
        case .backend(_, let message):
            return message
        case .emptyAnswer:
            return "Claude returned an empty answer."
        }
    }
}

struct ClaudeBackendClient: Sendable {
    private struct Request: Encodable {
        let id: String
        let question: String
        let model: String
    }

    private struct Response: Decodable {
        struct Failure: Decodable {
            let code: String
            let message: String
        }

        let id: String
        let answer: String?
        let error: Failure?
    }

    private let configuration: ClaudeBackendConfiguration

    init(configuration: ClaudeBackendConfiguration = .bundled) {
        self.configuration = configuration
    }

    func ask(_ question: String) async throws -> String {
        try await withThrowingTaskGroup(of: String.self) { group in
            group.addTask {
                try await run(question)
            }
            group.addTask {
                try await Task.sleep(nanoseconds: UInt64(configuration.timeout * 1_000_000_000))
                throw ClaudeBackendError.timeout
            }

            guard let answer = try await group.next() else {
                throw ClaudeBackendError.malformedResponse
            }
            group.cancelAll()
            return answer
        }
    }

    private func run(_ question: String) async throws -> String {
        let fileManager = FileManager.default
        guard fileManager.isExecutableFile(atPath: configuration.runtimeURL.path) else {
            throw ClaudeBackendError.missingRuntime
        }
        guard fileManager.fileExists(atPath: configuration.scriptURL.path) else {
            throw ClaudeBackendError.missingScript
        }
        guard fileManager.fileExists(atPath: configuration.zshrcURL.path) else {
            throw ClaudeBackendError.missingZshrc
        }

        let identifier = UUID().uuidString
        let request = Request(id: identifier, question: question, model: configuration.model)
        var requestData = try JSONEncoder().encode(request)
        requestData.append(0x0A)

        let process = Process()
        let inputPipe = Pipe()
        let outputPipe = Pipe()
        let errorPipe = Pipe()
        process.executableURL = configuration.shellURL
        process.arguments = ["-dfc", Self.shellCommand]
        process.standardInput = inputPipe
        process.standardOutput = outputPipe
        process.standardError = errorPipe
        process.environment = Self.processEnvironment(configuration: configuration)

        return try await withTaskCancellationHandler {
            do {
                try process.run()
            } catch {
                throw ClaudeBackendError.launchFailed(error.localizedDescription)
            }

            inputPipe.fileHandleForWriting.write(requestData)
            try? inputPipe.fileHandleForWriting.close()

            async let outputData = outputPipe.fileHandleForReading.readToEnd()
            async let errorData = errorPipe.fileHandleForReading.readToEnd()
            let terminationStatus = await Task.detached {
                process.waitUntilExit()
                return process.terminationStatus
            }.value
            let stdout = try await outputData ?? Data()
            let stderr = try await errorData ?? Data()

            guard let line = String(data: stdout, encoding: .utf8)?
                .split(whereSeparator: \ .isNewline)
                .first,
                  let responseData = String(line).data(using: .utf8),
                  let response = try? JSONDecoder().decode(Response.self, from: responseData),
                  response.id == identifier else {
                if terminationStatus != 0,
                   let detail = String(data: stderr, encoding: .utf8)?
                    .trimmingCharacters(in: .whitespacesAndNewlines),
                   !detail.isEmpty {
                    throw ClaudeBackendError.launchFailed(String(detail.prefix(400)))
                }
                throw ClaudeBackendError.malformedResponse
            }

            if let failure = response.error {
                throw ClaudeBackendError.backend(code: failure.code, message: failure.message)
            }
            guard let answer = response.answer?.trimmingCharacters(in: .whitespacesAndNewlines),
                  !answer.isEmpty else {
                throw ClaudeBackendError.emptyAnswer
            }
            return answer
        } onCancel: {
            if process.isRunning {
                process.terminate()
            }
        }
    }

    private static func processEnvironment(
        configuration: ClaudeBackendConfiguration
    ) -> [String: String] {
        var environment = ProcessInfo.processInfo.environment
        environment["GENO_BACKEND_RUNTIME"] = configuration.runtimeURL.path
        environment["GENO_BACKEND_SCRIPT"] = configuration.scriptURL.path
        environment["GENO_ZSHRC_PATH"] = configuration.zshrcURL.path
        return environment
    }

    private static let shellCommand = #"""
source "$GENO_ZSHRC_PATH" >/dev/null 2>&1 || exit 72
exec "$GENO_BACKEND_RUNTIME" "$GENO_BACKEND_SCRIPT"
"""#
}
