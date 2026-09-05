import Foundation

struct AppVersion: Equatable {
    let shortVersion: String
    let buildNumber: String

    var label: String {
        "v\(shortVersion) (\(buildNumber))"
    }

    var applicationName: String {
        "GenoVoice \(label)"
    }

    static var current: AppVersion {
        let info = Bundle.main.infoDictionary ?? [:]
        return AppVersion(
            shortVersion: info["CFBundleShortVersionString"] as? String ?? "dev",
            buildNumber: info["CFBundleVersion"] as? String ?? "local"
        )
    }
}
