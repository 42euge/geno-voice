import AppKit

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    private var statusItem: NSStatusItem?
    private var hotKey: GlobalHotKey?
    private var overlayController: OverlayWindowController?
    private var settingsController: SettingsWindowController?
    private var coordinator: AgentCoordinator?

    func applicationDidFinishLaunching(_ notification: Notification) {
        SpeechPreferences.registerDefaults()
        let coordinator = AgentCoordinator()
        let overlayController = OverlayWindowController(model: coordinator)
        coordinator.onPresent = { [weak overlayController] in
            overlayController?.present()
        }
        coordinator.onDismiss = { [weak overlayController] in
            overlayController?.dismiss()
        }

        self.coordinator = coordinator
        self.overlayController = overlayController
        self.settingsController = SettingsWindowController()
        installMenuBarItem()

        let hotKey = GlobalHotKey { [weak coordinator] in
            coordinator?.toggleFromShortcut()
        }
        self.hotKey = hotKey
        let status = hotKey.registerOptionSpace()
        if status != noErr {
            coordinator.presentError(
                "Option–Space is already in use. Quit the other app using it, then relaunch GenoVoice."
            )
        }
    }

    private func installMenuBarItem() {
        let item = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)
        if let button = item.button {
            button.image = NSImage(
                systemSymbolName: "waveform.and.mic",
                accessibilityDescription: "GenoVoice"
            )
            button.toolTip = "\(AppVersion.current.applicationName) — Option–Space"
        }

        let menu = NSMenu()
        let askItem = NSMenuItem(
            title: "Ask a Quick Question   ⌥Space",
            action: #selector(askQuestion),
            keyEquivalent: ""
        )
        askItem.target = self
        menu.addItem(askItem)

        let settingsItem = NSMenuItem(
            title: "Settings…",
            action: #selector(openSettings),
            keyEquivalent: ","
        )
        settingsItem.keyEquivalentModifierMask = [.command]
        settingsItem.target = self
        menu.addItem(settingsItem)
        menu.addItem(.separator())

        let quitItem = NSMenuItem(
            title: "Quit \(AppVersion.current.applicationName)",
            action: #selector(quit),
            keyEquivalent: "q"
        )
        quitItem.keyEquivalentModifierMask = [.command]
        quitItem.target = self
        menu.addItem(quitItem)

        item.menu = menu
        statusItem = item
    }

    @objc private func askQuestion() {
        coordinator?.startQuestion()
    }

    @objc private func openSettings() {
        settingsController?.show()
    }

    @objc private func quit() {
        NSApplication.shared.terminate(nil)
    }
}
