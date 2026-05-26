"""Word Error Rate computation — pure string utility.

iter-105: closes the metric-taxonomy infrastructure for WER (1.6).
This module exposes the standard word-level edit-distance WER
formula:

    WER = (S + D + I) / N

where S = substitutions, D = deletions, I = insertions, N = number
of words in the reference. Uses dynamic-programming word-level
Levenshtein, no external dependencies (jiwer/python-Levenshtein
NOT pulled in — keeps the install footprint flat).

The function works on whitespace-tokenized words after light
normalization (lowercase + strip punctuation). Designed to match
the reporting-grade definition most STT vendors use.

The audio-fixture corpus that POPULATES this metric in production
is deferred to a future iteration. Wiring exists so a follow-up
can plug in real reference transcripts without touching this
module.
"""

from __future__ import annotations

import re

# Punctuation we strip so "hello!" and "hello" compare equal.
# Apostrophes are kept so "don't" stays "don't" — splitting on
# them inflates the word count and over-penalizes contractions.
_PUNCT_RE = re.compile(r"[^\w\s']")


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    cleaned = _PUNCT_RE.sub(" ", text.lower())
    return cleaned.split()


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate.

    Returns a float ``≥ 0.0``. Common ranges:
      - 0.0   — perfect transcription
      - 0.1   — production-grade STT (Whisper large)
      - 0.2-0.3 — production-grade STT on noisy audio
      - 1.0   — every reference word missed (hypothesis empty
                or completely wrong)
      - >1.0  — possible when insertions dominate (hypothesis
                much longer than reference, all wrong)

    Edge cases:
      - Empty reference + empty hypothesis: returns 0.0 (no
        errors against no words).
      - Empty reference + non-empty hypothesis: returns float
        equal to the hypothesis word count (every hyp word is
        an insertion against an empty reference). This is the
        standard convention; some libraries return inf.
      - Empty hypothesis + non-empty reference: returns 1.0
        (every reference word is a deletion).

    Implementation: classic word-level Levenshtein DP, O(R*H)
    time and space where R = |reference| and H = |hypothesis|
    in word counts. For the typical TurnMetrics scale (one
    utterance = 5-50 words), this is sub-millisecond per call.
    """
    ref = _tokenize(reference)
    hyp = _tokenize(hypothesis)

    n = len(ref)
    if n == 0:
        # No reference words. Return raw hyp length as a float —
        # matches the "every hyp word is an insertion" reading.
        # (Convention varies; we pick the float-rate version
        # rather than inf so callers don't trip on Inf math.)
        return float(len(hyp))

    if not hyp:
        # No hypothesis at all: every reference word is a deletion.
        return 1.0

    # DP table: dp[i][j] = edits to align ref[:i] with hyp[:j].
    dp = [[0] * (len(hyp) + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(len(hyp) + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,        # deletion
                dp[i][j - 1] + 1,        # insertion
                dp[i - 1][j - 1] + cost  # substitution / match
            )

    return dp[n][len(hyp)] / n
