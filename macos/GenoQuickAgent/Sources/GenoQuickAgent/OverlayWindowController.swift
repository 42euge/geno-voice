import AppKit
import Combine
import SwiftUI

final class QuickAgentPanel: NSPanel {
    override var canBecomeKey: Bool { true }
    override var canBecomeMain: Bool { false }
}

@MainActor
final class OverlayWindowController {
    private let model: AgentCoordinator
    private let panel: QuickAgentPanel
    private let hostingController: NSHostingController<OverlayView>
    private var modelSubscription: AnyCancellable?
    private var keyMonitor: Any?

    init(model: AgentCoordinator) {
        self.model = model
        panel = QuickAgentPanel(
            contentRect: NSRect(origin: .zero, size: OverlayLayout.preferredSize(for: .listening)),
            styleMask: [.borderless, .nonactivatingPanel],
            backing: .buffered,
            defer: false
        )
        panel.isOpaque = false
        panel.backgroundColor = .clear
        panel.hasShadow = false
        panel.level = .statusBar
        panel.hidesOnDeactivate = false
        panel.isReleasedWhenClosed = false
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .transient]
        hostingController = NSHostingController(rootView: OverlayView(model: model))
        OverlayWindowPolicy.lock(
            panel: panel,
            hostingController: hostingController,
            to: OverlayLayout.preferredSize(for: .listening)
        )
        panel.contentViewController = hostingController

        modelSubscription = model.objectWillChange.sink { [weak self] in
            DispatchQueue.main.async {
                self?.syncWithModel(animated: true)
            }
        }
        keyMonitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { [weak self] event in
            guard let self, self.panel.isVisible else { return event }
            if event.keyCode == 53 {
                self.model.cancelAndDismiss()
                return nil
            }
            if event.modifierFlags.contains(.command),
               event.charactersIgnoringModifiers?.lowercased() == "c",
               self.model.phase == .answer {
                self.model.copyAnswer()
                return nil
            }
            return event
        }
    }

    deinit {
        if let keyMonitor { NSEvent.removeMonitor(keyMonitor) }
    }

    func present() {
        syncWithModel(animated: false)
        panel.makeKeyAndOrderFront(nil)
        panel.orderFrontRegardless()
    }

    func dismiss() {
        panel.orderOut(nil)
    }

    private func syncWithModel(animated: Bool) {
        guard model.phase != .hidden else {
            dismiss()
            return
        }
        let screen = screenUnderPointer() ?? NSScreen.main
        guard let visibleFrame = screen?.visibleFrame else { return }
        let newFrame = OverlayLayout.frame(
            preferredSize: OverlayLayout.preferredSize(for: model.phase),
            within: visibleFrame
        )
        OverlayWindowPolicy.lock(
            panel: panel,
            hostingController: hostingController,
            to: newFrame.size
        )
        panel.setFrame(newFrame, display: true, animate: animated)
    }

    private func screenUnderPointer() -> NSScreen? {
        let pointer = NSEvent.mouseLocation
        return NSScreen.screens.first { NSMouseInRect(pointer, $0.frame, false) }
    }
}
