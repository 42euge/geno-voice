// swift-tools-version: 5.10

import PackageDescription

let package = Package(
    name: "GenoQuickAgent",
    platforms: [.macOS(.v13)],
    products: [
        .executable(name: "GenoQuickAgent", targets: ["GenoQuickAgent"]),
    ],
    targets: [
        .executableTarget(
            name: "GenoQuickAgent",
            linkerSettings: [
                .linkedFramework("AppKit"),
                .linkedFramework("AVFoundation"),
                .linkedFramework("Carbon"),
                .linkedFramework("Speech"),
                .linkedFramework("SwiftUI"),
            ]
        ),
    ]
)
