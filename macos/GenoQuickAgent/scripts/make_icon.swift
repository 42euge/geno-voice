#!/usr/bin/env swift

import AppKit
import Foundation

guard CommandLine.arguments.count == 2 else {
    fputs("usage: make_icon.swift OUTPUT.iconset\n", stderr)
    exit(2)
}

let outputDirectory = URL(fileURLWithPath: CommandLine.arguments[1], isDirectory: true)
try FileManager.default.createDirectory(at: outputDirectory, withIntermediateDirectories: true)

let icons: [(String, Int)] = [
    ("icon_16x16.png", 16),
    ("icon_16x16@2x.png", 32),
    ("icon_32x32.png", 32),
    ("icon_32x32@2x.png", 64),
    ("icon_128x128.png", 128),
    ("icon_128x128@2x.png", 256),
    ("icon_256x256.png", 256),
    ("icon_256x256@2x.png", 512),
    ("icon_512x512.png", 512),
    ("icon_512x512@2x.png", 1024),
]

func renderIcon(pixels: Int) throws -> Data {
    guard let bitmap = NSBitmapImageRep(
        bitmapDataPlanes: nil,
        pixelsWide: pixels,
        pixelsHigh: pixels,
        bitsPerSample: 8,
        samplesPerPixel: 4,
        hasAlpha: true,
        isPlanar: false,
        colorSpaceName: .deviceRGB,
        bytesPerRow: 0,
        bitsPerPixel: 0
    ) else {
        throw NSError(domain: "GenoQuickAgentIcon", code: 1)
    }

    NSGraphicsContext.saveGraphicsState()
    defer { NSGraphicsContext.restoreGraphicsState() }
    NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: bitmap)
    NSGraphicsContext.current?.imageInterpolation = .high

    let scale = CGFloat(pixels) / 1024
    let canvas = NSRect(x: 0, y: 0, width: pixels, height: pixels)
    NSColor.clear.setFill()
    canvas.fill()

    let tile = NSBezierPath(
        roundedRect: canvas.insetBy(dx: 80 * scale, dy: 80 * scale),
        xRadius: 225 * scale,
        yRadius: 225 * scale
    )
    let gradient = NSGradient(colors: [
        NSColor(calibratedRed: 0.58, green: 0.45, blue: 1, alpha: 1),
        NSColor(calibratedRed: 0.25, green: 0.13, blue: 0.71, alpha: 1),
    ])!
    gradient.draw(in: tile, angle: -48)

    let highlight = NSBezierPath(
        roundedRect: canvas.insetBy(dx: 86 * scale, dy: 86 * scale),
        xRadius: 218 * scale,
        yRadius: 218 * scale
    )
    NSColor.white.withAlphaComponent(0.15).setStroke()
    highlight.lineWidth = 10 * scale
    highlight.stroke()

    let barHeights: [CGFloat] = [150, 285, 410, 235, 355, 175]
    let barWidth = 56 * scale
    let gap = 38 * scale
    let totalWidth = CGFloat(barHeights.count) * barWidth + CGFloat(barHeights.count - 1) * gap
    var x = (CGFloat(pixels) - totalWidth) / 2
    for height in barHeights {
        let rect = NSRect(
            x: x,
            y: (CGFloat(pixels) - height * scale) / 2,
            width: barWidth,
            height: height * scale
        )
        let bar = NSBezierPath(roundedRect: rect, xRadius: barWidth / 2, yRadius: barWidth / 2)
        NSColor.white.withAlphaComponent(0.94).setFill()
        bar.fill()
        x += barWidth + gap
    }

    guard let data = bitmap.representation(using: .png, properties: [:]) else {
        throw NSError(domain: "GenoQuickAgentIcon", code: 2)
    }
    return data
}

for (filename, pixels) in icons {
    try renderIcon(pixels: pixels).write(to: outputDirectory.appendingPathComponent(filename))
}
