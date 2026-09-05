import AppKit

@main
struct GenoQuickAgentMain {
    @MainActor
    static func main() {
        let application = NSApplication.shared
        let applicationDelegate = AppDelegate()

        application.delegate = applicationDelegate
        application.setActivationPolicy(.accessory)
        application.run()
    }
}
