import AppKit

enum OverlayLayout {
    static let screenMargin: CGFloat = 16
    static let preferredBottomOffset: CGFloat = 78

    static func preferredSize(for phase: AgentPhase) -> NSSize {
        switch phase {
        case .listening:
            return NSSize(width: 430, height: 82)
        case .transcribing, .thinking:
            return NSSize(width: 480, height: 92)
        case .answer:
            return NSSize(width: 620, height: 360)
        case .error:
            return NSSize(width: 460, height: 138)
        case .hidden:
            return NSSize(width: 430, height: 82)
        }
    }

    static func frame(
        preferredSize: NSSize,
        within visibleFrame: NSRect,
        margin: CGFloat = screenMargin,
        bottomOffset: CGFloat = preferredBottomOffset
    ) -> NSRect {
        let horizontalInset = min(margin, visibleFrame.width / 2)
        let verticalInset = min(margin, visibleFrame.height / 2)
        let availableWidth = max(0, visibleFrame.width - horizontalInset * 2)
        let availableHeight = max(0, visibleFrame.height - verticalInset * 2)
        let width = min(preferredSize.width, availableWidth)
        let height = min(preferredSize.height, availableHeight)

        let minimumX = visibleFrame.minX + horizontalInset
        let maximumX = visibleFrame.maxX - horizontalInset - width
        let centeredX = visibleFrame.midX - width / 2
        let x = min(max(centeredX, minimumX), maximumX)

        let minimumY = visibleFrame.minY + verticalInset
        let maximumY = visibleFrame.maxY - verticalInset - height
        let preferredY = visibleFrame.minY + bottomOffset
        let y = min(max(preferredY, minimumY), maximumY)

        return NSRect(x: x, y: y, width: width, height: height)
    }
}
