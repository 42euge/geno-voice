# GenoVoice for macOS

GenoVoice is a small menu-bar utility for one-off LLM questions. Press
Option–Space from any app, speak, and pause. The recording capsule turns into a
concise answer without opening a chat window.

## Build the installable DMG

Build requirements: macOS 13 or newer, Xcode command-line tools, Node/npm, and
network access on the first build. The finished app bundles its own runtime;
people installing the DMG do not need these development tools. Speech input
requires a Mac that supports on-device recognition for English.

```bash
cd macos/GenoQuickAgent
./scripts/build_dmg.sh
```

The current build produces:

- `dist/GenoVoice v0.2.1 (10).app`
- `dist/GenoVoice v0.2.1 (10).dmg`

The app name and every overlay header carry the same semantic version and build
number. `dist/GenoVoice.dmg` is also refreshed as a stable latest-build
alias.

Open the DMG and drag the app to Applications. A local build is ad-hoc signed,
so macOS may require Control-click → Open the first time. For distribution,
provide a Developer ID identity before running the same script:

```bash
CODESIGN_IDENTITY="Developer ID Application: Your Name (TEAMID)" ./scripts/build_dmg.sh
```

Apple notarization is still required before sharing the DMG broadly.

## Native speech playback

Completed answers use macOS native text-to-speech, entirely on-device and with
no additional model download or permission. Auto-speak is enabled by default.
The menu-bar Settings window lets you disable auto-speak, choose System Default
or an installed English voice, and adjust the speaking rate.

During playback, the answer highlights the word currently being spoken. The
answer card's Speak action replays an answer from the beginning and changes to
Stop while speech is active. Dismissing the card, asking another question, or
showing an error stops playback and clears the highlight.

## Claude SDK and BlueGPT

The DMG includes its own arm64 Node 22 runtime, Claude Agent SDK, and Claude
executable. The installed app does not require Ollama, Homebrew, Node, npm, or
Claude Code.

For each question, the app starts a one-turn, no-tools Sonnet agent and sources
your BlueGPT endpoint and credentials from `~/.zshrc`. Existing BlueGPT setups
using these variables are supported:

```bash
# Endpoint: either form is accepted. A trailing /v1 is removed when mapping
# OPENAI_BASE_URL for the Claude SDK.
export ANTHROPIC_BASE_URL="https://your-bluegpt-host"
# export OPENAI_BASE_URL="https://your-bluegpt-host/v1"

# Authentication: the first populated value is used.
export ANTHROPIC_AUTH_TOKEN="..."
# export BLUEGPT_API_TOKEN="..."
# export BLUEGPT_API_KEY="..."
# export OPENAI_API_KEY="..."
```

Restart GenoVoice after changing `~/.zshrc`. The waveform menu-bar
icon → Settings shows the active backend, model, and configuration source. The
app never stores credentials in macOS Keychain.

Audio is handled by Apple's on-device speech recognizer. Only the transcribed
question is sent to BlueGPT; the agent has no tools and receives no conversation
history.

## Interaction

- Option–Space while hidden: start listening.
- Pause for about one second: submit automatically.
- Option–Space while any overlay is visible: dismiss it.
- Speak/Stop on an answer: replay it from the beginning or stop playback.
- Command–C on an answer: copy it.
- Escape: close the overlay.

If microphone or speech-recognition access has been denied, the error card
offers a button that opens the exact Privacy & Security pane. Enable Geno Quick
Agent there, then press Option–Space to try again.

If the app reports missing configuration, confirm that `~/.zshrc` exists and
sets one endpoint and one authentication variable from the list above. If the
backend is missing, reinstall the app from the versioned DMG. A one-minute
timeout usually means the BlueGPT endpoint is unreachable or busy.

GenoVoice has no Dock icon or main window. Settings and Quit live under
the waveform menu-bar icon. If Option–Space is already owned by Superwhisper or
another utility, quit that app and relaunch GenoVoice.
