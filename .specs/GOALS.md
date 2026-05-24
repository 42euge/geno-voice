# Goals

Concrete targets toward the vision. Review monthly.

## Active

### G1 — End-to-end latency < 500ms
Measure mic-input to speaker-output on Apple Silicon (M1+). Includes VAD → STT → agent round-trip → TTS first-byte. Current baseline: unmeasured — instrument it, then optimize.

#### G1a — STT benchmarking system
Pytest-based benchmark suite comparing engine variants on standard audio. Measures latency (median/p95), real-time factor, WER. VS Code Test Explorer integrated — run individual engines or full sweeps from the sidebar.

### G2 — Claude Code voice integration
Ship a working mode where you speak commands into a Claude Code session and hear responses. Minimum: STT → paste to stdin, TTS reads stdout. Stretch: interrupt-aware, streaming.

### G3 — Reliable turn-taking
The turn-taking engine (`session/turn_taking.py`) needs to handle real conversations — overlapping speech, false starts, long pauses that aren't endings. Validate against recorded sessions, not just unit tests.

### G4 — TTS streaming playback
Kokoro currently generates full audio then plays. Switch to chunked streaming — first audio chunk plays while the rest generates. Critical for perceived latency.

### G5 — Packaging as a reusable SDK
Other geno-* projects (geno-reflect) already import this. Formalize: stable Python API, versioned, pip-installable. Document the integration surface.

### G6 — Onboard additional STT engines
Expand beyond Whisper/Gemma4. Candidates: NVIDIA Parakeet (NeMo), whisper.cpp (via whisper-cpp-python), distil-whisper. Each plugs into the `STTEngine` base class and gets benchmarked head-to-head.

## Completed

_(none yet)_

## Deferred

### D1 — Wake word detection
"Hey Geno" style activation. Useful but not blocking — push-to-talk and VAD-based activation cover current use cases.

### D2 — Multi-speaker diarization in real-time
Offline diarization exists in examples. Real-time is a harder problem; defer until single-speaker pipeline is solid.
