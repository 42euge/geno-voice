// iter-193 — node:test suite for client/voice-capture.js.
//
// The repo's first browser-client unit tests. The ContinuousListener VAD
// state machine is driven directly through `_handleFrame` (no AudioContext,
// no microphone) by stubbing `performance.now()` to a controllable clock and
// feeding synthetic frames whose constant sample value sets the frame RMS.
//
// Focus: the iter-193 pre-roll ring buffer that prepends pre-onset audio to a
// committed speech segment so the quiet soft attack the RMS gate clips is
// recovered. Mirrors the replay-harness model proven in iter-191
// (fixtures/replay_vad.py) — default prerollMs=0 must be exact parity with the
// historical clip-the-opening behaviour.

import test from "node:test";
import assert from "node:assert/strict";
import {
  ContinuousListener,
  mergeFloat32Chunks,
  downsampleBuffer,
  encodeWav,
} from "./voice-capture.js";

const FRAME = 1024; // samples per frame (matches the client's worklet hop)
const SR = 48000; // default sample rate when no AudioContext is present
const FRAME_MS = (FRAME / SR) * 1000; // ~21.33ms per frame

// A constant-value frame has RMS == |value|, so `value` directly chooses
// whether the frame sits over or under `silenceThreshold`.
function frame(value, n = FRAME) {
  return new Float32Array(n).fill(value);
}

// Drive a listener through a sequence of frames with a controllable clock.
// Each frame advances `performance.now()` by FRAME_MS so the onset debounce
// (>200ms) behaves like real time. Restores the global clock afterwards.
function feed(listener, frames) {
  const orig = performance.now;
  // Start at a positive offset: the client uses truthiness on the candidate
  // timestamp (`!this._speechCandidate`), so a literal 0 clock would read as
  // "no candidate". Real performance.now() is never 0.
  let clock = 1000;
  performance.now = () => clock;
  try {
    listener.active = true;
    listener._muted = false;
    for (const f of frames) {
      listener._handleFrame(f);
      clock += FRAME_MS;
    }
  } finally {
    performance.now = orig;
  }
}

// Frames over threshold long enough to commit: the candidate must hold for
// >200ms, i.e. >~10 frames at 21.33ms each. Use 14 to clear it comfortably.
const COMMIT_FRAMES = 14;

function newListener(opts = {}) {
  return new ContinuousListener({ silenceThreshold: 0.015, ...opts });
}

test("default prerollMs is 0 and capacity is 0 (zero-cost path)", () => {
  const l = newListener();
  assert.equal(l.prerollMs, 0);
  assert.equal(l._prerollMaxSamples, 0);
  // _pushPreroll is a no-op when disabled.
  l._pushPreroll(frame(0.0001));
  assert.equal(l._prerollChunks.length, 0);
  assert.deepEqual(l._drainPreroll(), []);
});

test("parity: with prerollMs=0 a committed segment holds only candidate frames", () => {
  const l = newListener({ prerollMs: 0 });
  // 5 quiet pre-onset frames, then loud frames that commit to speaking.
  const frames = [
    ...Array.from({ length: 5 }, () => frame(0.0001)),
    ...Array.from({ length: COMMIT_FRAMES }, () => frame(0.5)),
  ];
  feed(l, frames);
  assert.equal(l.speaking, true);
  // No pre-onset frames prepended — segment starts at the candidate.
  assert.equal(l.chunks.length, COMMIT_FRAMES);
  // Every retained frame is a loud one (rms ~0.5), none of the quiet ramp.
  for (const c of l.chunks) {
    const rms = Math.sqrt(c.reduce((s, v) => s + v * v, 0) / c.length);
    assert.ok(rms > 0.4, `expected loud frame, got rms=${rms}`);
  }
});

test("prerollMs>0 prepends pre-onset frames to the committed segment", () => {
  const l = newListener({ prerollMs: 100 }); // 100ms ≈ 4800 samples ≈ ~5 frames
  const quiet = Array.from({ length: 5 }, () => frame(0.0001));
  const loud = Array.from({ length: COMMIT_FRAMES }, () => frame(0.5));
  feed(l, [...quiet, ...loud]);
  assert.equal(l.speaking, true);
  // The segment now reaches back across the buffered quiet frames.
  assert.ok(
    l.chunks.length > COMMIT_FRAMES,
    `expected pre-roll prepended, got ${l.chunks.length} frames`,
  );
  // The earliest retained frames are the quiet ramp-up (the recovered attack).
  const firstRms = Math.sqrt(
    l.chunks[0].reduce((s, v) => s + v * v, 0) / l.chunks[0].length,
  );
  assert.ok(firstRms < 0.01, `expected quiet leading frame, got rms=${firstRms}`);
});

test("pre-roll ring is bounded to its sample capacity", () => {
  const l = newListener({ prerollMs: 50 }); // 50ms ≈ 2400 samples ≈ ~2-3 frames
  // Push far more quiet frames than the ring can hold.
  for (let i = 0; i < 50; i++) l._pushPreroll(frame(0.0001));
  // Retained audio never exceeds capacity by more than one frame's slack.
  assert.ok(
    l._prerollSamples <= l._prerollMaxSamples + FRAME,
    `ring overflow: ${l._prerollSamples} > ${l._prerollMaxSamples}+${FRAME}`,
  );
  assert.ok(l._prerollChunks.length >= 1);
  // Internal accounting stays consistent with the retained chunks.
  const actual = l._prerollChunks.reduce((s, c) => s + c.length, 0);
  assert.equal(l._prerollSamples, actual);
});

test("a broken speech candidate folds into the pre-roll ring", () => {
  const l = newListener({ prerollMs: 200 });
  const orig = performance.now;
  let clock = 1000; // positive: candidate timestamp is truthiness-checked
  performance.now = () => clock;
  try {
    l.active = true;
    l._muted = false;
    // A few loud frames that do NOT hold long enough to commit (< 200ms)...
    for (let i = 0; i < 3; i++) {
      l._handleFrame(frame(0.5));
      clock += FRAME_MS;
    }
    assert.equal(l.speaking, false);
    assert.ok(l._candidateChunks); // candidate pending
    // ...then a quiet frame breaks the candidate; its frames become history.
    l._handleFrame(frame(0.0001));
    clock += FRAME_MS;
    assert.equal(l._candidateChunks, null);
    // The 3 candidate frames + 1 quiet frame all went to the ring.
    assert.equal(l._prerollChunks.length, 4);
  } finally {
    performance.now = orig;
  }
});

test("drained pre-roll resets the ring so the next onset starts clean", () => {
  const l = newListener({ prerollMs: 100 });
  for (let i = 0; i < 3; i++) l._pushPreroll(frame(0.0001));
  assert.ok(l._prerollChunks.length > 0);
  const drained = l._drainPreroll();
  assert.equal(drained.length, 3);
  assert.equal(l._prerollChunks.length, 0);
  assert.equal(l._prerollSamples, 0);
});

test("_recomputePreroll scales capacity with sample rate", () => {
  const l = newListener({ prerollMs: 100 });
  assert.equal(l._prerollMaxSamples, Math.round(0.1 * SR)); // 4800 @ 48k
  l._sampleRate = 16000;
  l._recomputePreroll();
  assert.equal(l._prerollMaxSamples, 1600); // 100ms @ 16k
});

test("stop() clears the pre-roll ring", async () => {
  const l = newListener({ prerollMs: 100 });
  for (let i = 0; i < 3; i++) l._pushPreroll(frame(0.0001));
  assert.ok(l._prerollChunks.length > 0);
  await l.stop();
  assert.equal(l._prerollChunks.length, 0);
  assert.equal(l._prerollSamples, 0);
});

// Sanity coverage of the pure audio helpers (also previously untested in JS).
test("mergeFloat32Chunks concatenates in order", () => {
  const merged = mergeFloat32Chunks([
    new Float32Array([1, 2]),
    new Float32Array([3]),
    new Float32Array([4, 5]),
  ]);
  assert.deepEqual([...merged], [1, 2, 3, 4, 5]);
});

test("downsampleBuffer halves length and is identity at equal rates", () => {
  const input = new Float32Array([0, 1, 0, 1, 0, 1, 0, 1]);
  assert.strictEqual(downsampleBuffer(input, 16000, 16000), input);
  const down = downsampleBuffer(input, 32000, 16000);
  assert.equal(down.length, 4);
});

test("encodeWav writes a 44-byte RIFF/WAVE header", () => {
  const wav = encodeWav(new Float32Array([0, 0.5, -0.5]), 16000);
  const ascii = (off, len) =>
    String.fromCharCode(...wav.subarray(off, off + len));
  assert.equal(ascii(0, 4), "RIFF");
  assert.equal(ascii(8, 4), "WAVE");
  assert.equal(ascii(36, 4), "data");
  assert.equal(wav.length, 44 + 3 * 2); // header + 3 int16 samples
});
