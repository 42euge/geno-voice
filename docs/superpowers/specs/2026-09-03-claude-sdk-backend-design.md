# GenoVoice Claude SDK Backend Design

## Goal

Replace the direct OpenAI-compatible HTTP call with a fully bundled local
backend that uses the Claude Agent SDK and routes requests through the user's
BlueGPT configuration from `~/.zshrc`.

## Architecture

The macOS app launches one backend subprocess per question. The subprocess is
started through `/bin/zsh`, explicitly sources `~/.zshrc`, maps the existing
BlueGPT variables to the Anthropic variables consumed by the Claude Agent SDK,
and then executes the Node runtime bundled inside the app. The Swift and Node
processes exchange one JSON object per line over standard input and output; no
localhost port is opened.

The Swift module presents one interface:

```swift
func ask(_ question: String) async throws -> String
```

Its implementation owns bundle-path resolution, subprocess lifecycle, JSON
encoding and decoding, timeouts, cancellation, bounded stderr capture, and
user-facing errors. The Node backend owns Claude Agent SDK configuration and
response extraction.

## Backend Policy

- Use `@anthropic-ai/claude-agent-sdk` pinned to `0.3.260`.
- Use the `sonnet` BlueGPT model alias.
- Allow one turn and no tools.
- Use the existing concise-answer system instruction.
- Do not persist sessions or conversation history.
- Never read or write macOS Keychain.
- Source credentials only from `~/.zshrc`.
- Prefer `BLUEGPT_API_TOKEN`, then `BLUEGPT_API_KEY`, then `OPENAI_API_KEY`.
- Prefer `ANTHROPIC_BASE_URL`; otherwise derive it from `OPENAI_BASE_URL` by
  removing a trailing `/v1`.

## Packaging

The app bundle contains `Contents/Resources/ClaudeBackend/` with:

- an arm64 standalone Node 22 executable;
- the backend JavaScript;
- production `node_modules`, including the Claude Agent SDK and its packaged
  Claude executable.

The build downloads the official Node 22.22.3 Darwin arm64 archive into a
local cache when a portable runtime is unavailable. The built DMG therefore
does not depend on Homebrew, NVM, npm, Node, Claude Code, or Ollama on the
destination Mac. macOS-provided `/bin/zsh` is the only external executable.

## UI and Errors

Settings identify the backend as “Claude SDK via BlueGPT” and explain that
endpoint and credentials come from `~/.zshrc`; API-key and base-URL fields are
removed. Errors distinguish missing configuration, backend launch failure,
timeout, malformed backend output, and Claude SDK failure. The compact overlay
and Option–Space dismissal behavior remain unchanged.

## Verification

- Node tests cover request validation, one-turn/no-tools SDK options, answer
  extraction, and safe error serialization.
- Swift regression tests exercise the real subprocess protocol using a tiny
  fake backend executable.
- Bundle tests require the Node executable, backend entrypoint, SDK package,
  and packaged Claude executable.
- An installed-build smoke test sends a harmless question through the bundled
  backend and the real BlueGPT configuration, asserting a non-empty answer.

## Release

This architecture change ships as `v0.2.0 (9)` and retains the stable bundle
identifier `com.geno.quickagent`.
