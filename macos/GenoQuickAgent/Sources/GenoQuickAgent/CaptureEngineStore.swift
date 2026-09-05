final class CaptureEngineStore<Engine> {
    private let makeEngine: () -> Engine
    private(set) var current: Engine?

    init(makeEngine: @escaping () -> Engine) {
        self.makeEngine = makeEngine
    }

    @discardableResult
    func begin() -> Engine {
        let engine = makeEngine()
        current = engine
        return engine
    }

    func end() {
        current = nil
    }
}
