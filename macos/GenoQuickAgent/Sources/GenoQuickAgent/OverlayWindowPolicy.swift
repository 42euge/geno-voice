import AppKit
import SwiftUI

@MainActor
enum OverlayWindowPolicy {
    static func lock<Content: View>(
        panel: NSPanel,
        hostingController: NSHostingController<Content>,
        to size: NSSize
    ) {
        hostingController.sizingOptions = []
        panel.isMovable = false
        panel.isMovableByWindowBackground = false
        panel.minSize = size
        panel.maxSize = size
        panel.contentMinSize = size
        panel.contentMaxSize = size
    }
}
