import Foundation

enum SpokenWordRange {
    static func range(_ range: NSRange?, in text: String) -> Range<String.Index>? {
        guard let range,
              range.location != NSNotFound,
              range.length > 0 else {
            return nil
        }
        return Range(range, in: text)
    }

    static func substring(_ range: NSRange?, in text: String) -> String? {
        guard let converted = Self.range(range, in: text) else { return nil }
        return String(text[converted])
    }
}
