# GenoVoice Claude SDK Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship GenoVoice with a bundled Claude Agent SDK backend that sources Anthropic-compatible configuration from `~/.zshrc` and requires no Keychain, Ollama, Node, or Claude Code installation.

**Architecture:** Swift launches a bundled Node backend through `/bin/zsh` for each question and exchanges one JSON request and response over standard input/output. The backend sources standard Claude Agent SDK environment variables, runs one no-tools Sonnet turn, and exits.

**Tech Stack:** Swift 5.10, AppKit/SwiftUI, Foundation `Process`, Node 22.22.3 arm64, `@anthropic-ai/claude-agent-sdk` 0.3.260, Node test runner, Bash packaging.

## Global Constraints

- macOS 13 or newer on Apple Silicon.
- Bundle identifier remains `com.geno.quickagent`.
- Backend uses `sonnet`, one turn, and no tools.
- Credentials come only from `~/.zshrc`; no Keychain access.
- DMG contains its own Node runtime and Claude Agent SDK.
- Overlay sizing and shortcut behavior must not regress.
- Release version is `v0.2.0 (9)`.

---

### Task 1: Claude SDK JSON-lines backend

**Files:**
- Create: `macos/GenoQuickAgent/Backend/package.json`
- Create: `macos/GenoQuickAgent/Backend/package-lock.json`
- Create: `macos/GenoQuickAgent/Backend/claude-backend.mjs`
- Create: `macos/GenoQuickAgent/Backend/claude-backend.test.mjs`

**Interfaces:**
- Consumes: one stdin line shaped as `{"id":"...","question":"...","model":"sonnet"}` and standard Anthropic environment variables.
- Produces: one stdout line shaped as `{"id":"...","answer":"..."}` or `{"id":"...","error":{"code":"...","message":"..."}}`.

- [ ] Write Node tests that inject a fake `query` function and assert the
  backend uses `maxTurns: 1`, `allowedTools: []`, model `sonnet`, and returns
  the final non-empty result.
- [ ] Run `node --test Backend/claude-backend.test.mjs` and verify it fails
  because the backend module does not exist.
- [ ] Implement request validation, SDK options, result extraction, and
  JSON-safe error responses in `claude-backend.mjs`.
- [ ] Run the Node tests and verify all cases pass.
- [ ] Generate and retain a lockfile pinned to Claude Agent SDK `0.3.260`.

### Task 2: Swift subprocess module

**Files:**
- Create: `macos/GenoQuickAgent/Sources/GenoQuickAgent/ClaudeBackendClient.swift`
- Modify: `macos/GenoQuickAgent/Sources/GenoQuickAgent/AgentCoordinator.swift`
- Modify: `macos/GenoQuickAgent/scripts/test_regressions.sh`
- Modify: `macos/GenoQuickAgent/Tests/Regression/main.swift`
- Delete: `macos/GenoQuickAgent/Sources/GenoQuickAgent/LLMClient.swift`
- Delete: `macos/GenoQuickAgent/Sources/GenoQuickAgent/KeychainStore.swift`

**Interfaces:**
- Consumes: `ClaudeBackendClient.ask(_ question: String) async throws -> String`.
- Produces: a non-empty answer or typed `ClaudeBackendError` suitable for the
  existing compact error overlay.

- [ ] Add a regression test that launches a real temporary fake backend,
  verifies the question crosses the JSON-lines protocol, and receives its
  literal answer.
- [ ] Run `scripts/test_regressions.sh` and verify the test fails because
  `ClaudeBackendClient` is absent.
- [ ] Implement bundle path resolution, explicit `.zshrc` sourcing, safe
  environment mapping, subprocess I/O, a 60-second timeout, cancellation,
  and typed response decoding.
- [ ] Replace `LLMClient` in `AgentCoordinator` with `ClaudeBackendClient` and
  keep request cancellation semantics.
- [ ] Run Swift regression tests and verify they pass.

### Task 3: Settings, version, and self-contained packaging

**Files:**
- Modify: `macos/GenoQuickAgent/Sources/GenoQuickAgent/SettingsView.swift`
- Delete: `macos/GenoQuickAgent/Sources/GenoQuickAgent/QuickAgentSettings.swift`
- Modify: `macos/GenoQuickAgent/Resources/Info.plist`
- Modify: `macos/GenoQuickAgent/README.md`
- Create: `macos/GenoQuickAgent/scripts/prepare_claude_backend.sh`
- Create: `macos/GenoQuickAgent/scripts/test_bundle_backend.sh`
- Modify: `macos/GenoQuickAgent/scripts/build_dmg.sh`

**Interfaces:**
- Consumes: official Node `v22.22.3` Darwin arm64 archive and locked npm dependencies.
- Produces: `Contents/Resources/ClaudeBackend/node`, `claude-backend.mjs`, and `node_modules` in the signed app bundle.

- [ ] Write a bundle test that fails unless the backend entrypoint, executable
  Node runtime, Claude Agent SDK package, and packaged Claude executable exist.
- [ ] Run it against the current app and verify the missing-backend failure.
- [ ] Implement cached official-Node acquisition, pinned SHA-256 verification,
  archive extraction with architecture validation, `npm ci --omit=dev`, and
  resource copying.
- [ ] Replace editable endpoint/key settings with read-only Claude SDK,
  `.zshrc`, and model information.
- [ ] Bump all version references to `v0.2.0 (9)`.
- [ ] Run the build and bundle tests and verify they pass.

### Task 4: Real backend and installed-app verification

**Files:**
- Modify: `macos/GenoQuickAgent/README.md`

**Interfaces:**
- Consumes: bundled backend plus the user's existing `~/.zshrc` Anthropic values.
- Produces: a non-empty Claude answer and an installed signed app/DMG.

- [ ] Send `{"id":"smoke","question":"Reply with only OK.","model":"sonnet"}`
  through the bundled backend and assert the response has a non-empty `answer`.
- [ ] Run all Swift, Node, signing-identity, microphone-entitlement, and bundled-backend tests.
- [ ] Build and verify the `GenoVoice v0.2.0 (9).dmg` checksum.
- [ ] Install only `GenoVoice v0.2.0 (9).app`, relaunch it, and verify
  the process remains alive with no overlay visible at startup.
- [ ] Update README build, configuration, privacy, and troubleshooting text to
  match the shipped Claude SDK backend.
