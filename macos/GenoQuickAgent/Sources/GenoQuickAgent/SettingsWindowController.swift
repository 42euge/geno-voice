import AppKit
import SwiftUI

@MainActor
final class SettingsWindowController: NSObject, NSWindowDelegate {
    private let window: NSWindow

    override init() {
        let hostingController = NSHostingController(rootView: SettingsView())
        window = NSWindow(contentViewController: hostingController)
        super.init()
        window.title = "\(AppVersion.current.applicationName) Settings"
        window.styleMask = [.titled, .closable, .miniaturizable]
        window.setContentSize(NSSize(width: 560, height: 605))
        window.isReleasedWhenClosed = false
        window.center()
        window.delegate = self
    }

    func show() {
        NSApp.activate(ignoringOtherApps: true)
        window.center()
        window.makeKeyAndOrderFront(nil)
    }
}
