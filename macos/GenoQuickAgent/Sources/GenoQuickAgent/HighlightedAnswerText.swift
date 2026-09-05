import Foundation
import SwiftUI

enum HighlightedAnswerText {
    static func activeWord(text: String, range: NSRange?) -> String? {
        SpokenWordRange.substring(range, in: text)
    }

    static func attributed(text: String, range: NSRange?) -> AttributedString {
        var answer = AttributedString(text)
        answer.foregroundColor = .white

        guard let spokenRange = SpokenWordRange.range(range, in: text),
              let lowerBound = AttributedString.Index(spokenRange.lowerBound, within: answer),
              let upperBound = AttributedString.Index(spokenRange.upperBound, within: answer) else {
            return answer
        }

        let activeRange = lowerBound..<upperBound
        answer[activeRange].foregroundColor = .white
        answer[activeRange].backgroundColor = Color(red: 0.38, green: 0.24, blue: 0.72)
        answer[activeRange].font = .system(size: 15, weight: .semibold)
        return answer
    }
}
