# GenoVoice Native TTS and Word Highlighting Design

## Goal

Make every completed answer optionally speak itself aloud, visibly highlight the
word currently being spoken, and replace the adaptive grey answer card with a
reliably dark, high-contrast surface.

## Scope

This is the basic functional release. It includes:

- native macOS text-to-speech with no new runtime or model download;
- automatic playback when an answer arrives, enabled by default;
- Settings controls for auto-speak, voice, and speaking rate;
- replay/stop control on the answer card;
- word-level highlighting synchronized to speech playback;
- playback cancellation whenever the card is dismissed, a new question starts,
  or an error replaces the answer;
- an explicit dark answer surface that does not adapt into the grey appearance
  shown by `ultraThinMaterial`;
- release version `v0.2.1 (10)` in the app name and UI.

Voice previews, volume controls, Kokoro/Piper bundling, downloadable voices,
sentence-level animation, and audio interruption/barge-in are deferred.

## Architecture

### Speech playback module

`SpeechPlaybackController` is a deep module around `AVSpeechSynthesizer`. Its
interface is limited to starting automatic playback, toggling manual playback,
stopping, and reporting speaking state plus the current `NSRange`.

The implementation owns `AVSpeechSynthesizerDelegate`, utterance construction,
voice lookup, rate clamping, callback normalization, and cancellation cleanup.
The seam is deliberately native-TTS-shaped today without exposing
`AVSpeechSynthesizer` to the coordinator or SwiftUI. A future Kokoro adapter can
replace the implementation without changing answer-state or rendering logic.

### Configuration

`SpeechPreferences` owns three `UserDefaults` values:

- `speech.autoSpeak`: `Bool`, default `true`;
- `speech.voiceIdentifier`: `String`, default empty for the system English
  voice;
- `speech.rate`: `Double`, default `0.48`, clamped to `0.35...0.60`.

Settings uses the same keys through `@AppStorage`. The voice menu shows “System
Default” plus installed English `AVSpeechSynthesisVoice` values. Changes apply
to the next automatic playback or replay; they do not interrupt speech already
in progress.

### Coordinator data flow

After the Claude backend returns a non-empty answer, `AgentCoordinator`:

1. publishes the answer and enters `.answer`;
2. asks the playback module to auto-speak if enabled;
3. publishes the playback module's `isSpeaking` and current UTF-16 `NSRange`;
4. clears both values when playback finishes or stops.

`cancelAndDismiss`, `startQuestion`, and `presentError` stop playback before
changing phase. A manual `toggleAnswerPlayback()` call replays from the start or
stops the current utterance.

### Highlighted answer rendering

`HighlightedAnswerText` converts the delegate's UTF-16 `NSRange` into Swift
string indices safely. The active word gets a purple background, white
foreground, and semibold emphasis; all other answer text remains high-contrast
white. Invalid, stale, or out-of-bounds ranges render plain text instead of
crashing.

The existing scroll view and text selection remain available. The answer card
adds one compact “Speak”/“Stop” action next to Copy and Ask another.

### Dark surface

The overlay stops using adaptive material. `OverlayPalette` supplies an opaque
near-black purple gradient, explicit light primary/secondary text, a subtle
border, and the existing shadow. Applying dark colors explicitly makes the card
stable across desktop wallpaper, inactive-window state, and system appearance.

## Error Handling

TTS is an enhancement to a successfully returned answer. If the selected voice
is unavailable, playback falls back to the system English voice. If synthesis
cannot begin or is interrupted, the answer remains visible and readable; the
playback state and highlight are cleared without replacing the answer with an
error card.

## Testing

- Pure regression tests cover preference defaults/rate clamping and UTF-16 range
  conversion, including emoji and invalid ranges.
- Coordinator regressions use a fake playback adapter to verify auto-speak,
  replay/stop, and stop-on-dismiss/new-question/error behavior.
- Palette regressions verify the explicit surface colors remain dark and opaque.
- Existing compact-overlay, off-screen layout, permission, shortcut-dismiss,
  Claude backend, packaging, signing, and entitlement tests remain green.
- Release verification manually renders an answer, confirms speech is audible,
  observes the active word moving during playback, and checks the answer card is
  dark rather than grey.

## Packaging and Privacy

`AVSpeechSynthesizer` is supplied by macOS, so the DMG remains self-contained and
does not grow a Python runtime or speech model. Answer text stays on the Mac for
speech synthesis. No new permission prompt, network request, or Keychain access
is introduced.
