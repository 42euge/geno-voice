const MIN_TRANSCRIBE_SECONDS = 0.35;

async function tryLoadWorklet(audioContext, workletUrl) {
  if (!workletUrl || !audioContext.audioWorklet) return false;
  try {
    await audioContext.audioWorklet.addModule(workletUrl);
    return true;
  } catch {
    return false;
  }
}

function mergeFloat32Chunks(chunks) {
  const totalLength = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const merged = new Float32Array(totalLength);
  let offset = 0;
  for (const chunk of chunks) {
    merged.set(chunk, offset);
    offset += chunk.length;
  }
  return merged;
}

function downsampleBuffer(input, sourceRate, targetRate) {
  if (sourceRate === targetRate) return input;
  if (sourceRate < targetRate) {
    throw new Error("Input sample rate is lower than the target sample rate.");
  }

  const ratio = sourceRate / targetRate;
  const outputLength = Math.round(input.length / ratio);
  const output = new Float32Array(outputLength);

  let outputIndex = 0;
  let inputIndex = 0;
  while (outputIndex < outputLength) {
    const nextInputIndex = Math.round((outputIndex + 1) * ratio);
    let sum = 0;
    let count = 0;
    for (let i = inputIndex; i < nextInputIndex && i < input.length; i++) {
      sum += input[i];
      count += 1;
    }
    output[outputIndex] = count > 0 ? sum / count : 0;
    outputIndex += 1;
    inputIndex = nextInputIndex;
  }

  return output;
}

function writeString(view, offset, value) {
  for (let i = 0; i < value.length; i++) {
    view.setUint8(offset + i, value.charCodeAt(i));
  }
}

function encodeWav(samples, sampleRate) {
  const pcmSamples = new Int16Array(samples.length);
  for (let i = 0; i < samples.length; i++) {
    const clamped = Math.max(-1, Math.min(1, samples[i]));
    pcmSamples[i] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;
  }

  const buffer = new ArrayBuffer(44 + pcmSamples.length * 2);
  const view = new DataView(buffer);

  writeString(view, 0, "RIFF");
  view.setUint32(4, 36 + pcmSamples.length * 2, true);
  writeString(view, 8, "WAVE");
  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(view, 36, "data");
  view.setUint32(40, pcmSamples.length * 2, true);

  let offset = 44;
  for (let i = 0; i < pcmSamples.length; i++) {
    view.setInt16(offset, pcmSamples[i], true);
    offset += 2;
  }

  return new Uint8Array(buffer);
}

export { mergeFloat32Chunks, downsampleBuffer, encodeWav };

export class VoiceRecorder {
  constructor({ onStateChange, workletUrl } = {}) {
    this.onStateChange = onStateChange;
    this._workletUrl = workletUrl || null;
    this.audioContext = null;
    this.inputNode = null;
    this.processorNode = null;
    this._workletNode = null;
    this.silentGain = null;
    this.stream = null;
    this.chunks = [];
    this.recording = false;
    this.startedAt = 0;
  }

  async start() {
    if (this.recording) return;
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error("Microphone capture is not available in this environment.");
    }

    const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextCtor) {
      throw new Error("The Web Audio API is not available.");
    }

    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
      },
    });

    this.audioContext = new AudioContextCtor();
    await this.audioContext.resume();

    this.inputNode = this.audioContext.createMediaStreamSource(this.stream);
    this.silentGain = this.audioContext.createGain();
    this.silentGain.gain.value = 0;
    this.chunks = [];

    if (await tryLoadWorklet(this.audioContext, this._workletUrl)) {
      this._workletNode = new AudioWorkletNode(this.audioContext, "voice-processor");
      this._workletNode.port.onmessage = (e) => {
        if (!this.recording) return;
        this.chunks.push(new Float32Array(e.data));
      };
      this.inputNode.connect(this._workletNode);
      this._workletNode.connect(this.silentGain);
    } else {
      this.processorNode = this.audioContext.createScriptProcessor(4096, 1, 1);
      this.processorNode.onaudioprocess = (event) => {
        if (!this.recording) return;
        this.chunks.push(new Float32Array(event.inputBuffer.getChannelData(0)));
      };
      this.inputNode.connect(this.processorNode);
      this.processorNode.connect(this.silentGain);
    }
    this.silentGain.connect(this.audioContext.destination);

    this.startedAt = performance.now();
    this.recording = true;
    if (this.onStateChange) this.onStateChange(true);
  }

  async stop() {
    if (!this.recording) return null;

    this.recording = false;
    if (this.onStateChange) this.onStateChange(false);

    const elapsedMs = performance.now() - this.startedAt;
    const sampleRate = this.audioContext ? this.audioContext.sampleRate : 48000;
    const chunks = this.chunks.slice();

    await this.cleanup();

    if (!chunks.length || elapsedMs < 200) return null;

    const merged = mergeFloat32Chunks(chunks);
    if (!merged.length) return null;

    const downsampled = downsampleBuffer(merged, sampleRate, 16000);
    if (downsampled.length < MIN_TRANSCRIBE_SECONDS * 16000) return null;
    return encodeWav(downsampled, 16000);
  }

  async cancel() {
    this.recording = false;
    if (this.onStateChange) this.onStateChange(false);
    await this.cleanup();
  }

  async cleanup() {
    if (this.inputNode) {
      this.inputNode.disconnect();
      this.inputNode = null;
    }
    if (this._workletNode) {
      this._workletNode.port.close();
      this._workletNode.disconnect();
      this._workletNode = null;
    }
    if (this.processorNode) {
      this.processorNode.disconnect();
      this.processorNode.onaudioprocess = null;
      this.processorNode = null;
    }
    if (this.silentGain) {
      this.silentGain.disconnect();
      this.silentGain = null;
    }
    if (this.stream) {
      for (const track of this.stream.getTracks()) track.stop();
      this.stream = null;
    }
    if (this.audioContext) {
      await this.audioContext.close();
      this.audioContext = null;
    }
  }
}

export class ContinuousListener {
  constructor({ onSpeechEnd, onStateChange, workletUrl, silenceThreshold = 0.015, silenceDurationMs = 800, minSpeechMs = 500, prerollMs = 0, gain = 1.0 } = {}) {
    this.onSpeechEnd = onSpeechEnd;
    this.onStateChange = onStateChange;
    this._workletUrl = workletUrl || null;
    this.silenceThreshold = silenceThreshold;
    this.silenceDurationMs = silenceDurationMs;
    this.minSpeechMs = minSpeechMs;
    // Software gain stage (iter-195): pre-amplify every captured frame before
    // RMS detection AND storage, mirroring the replay harness `gain` model
    // (fixtures/replay_vad.py `frame_rms`: `samples * gain`) whose threshold×
    // gain interaction the iter-192 grid characterised. A louder signal lifts
    // quiet far-field speech over a fixed RMS gate, so raising `gain` lets a
    // stricter `silenceThreshold` still catch it. 1.0 (the default) is an exact
    // no-op — the historical unity-gain path allocates nothing extra. Amplified
    // audio also flows into the committed segment, so STT sees the louder
    // signal too (a real upstream gainNode would do the same). Non-finite or
    // non-positive values fall back to unity.
    this.gain = Number.isFinite(gain) && gain > 0 ? gain : 1.0;
    // Pre-roll ring buffer (iter-193): keep the last `prerollMs` of pre-onset
    // audio so the committed segment recovers the quiet soft attack the RMS
    // gate would otherwise clip. 0 (the default) reproduces the historical
    // clip-the-opening behaviour exactly. Replay-proven safe in iter-191.
    this.prerollMs = prerollMs;
    this._sampleRate = 48000;
    this._prerollChunks = [];
    this._prerollSamples = 0;
    this._recomputePreroll();
    this.active = false;
    this.speaking = false;
    this.chunks = [];
    this.speechStartedAt = 0;
    this.lastSpeechAt = 0;
    this.silenceTimer = null;
    this.stream = null;
    this.audioContext = null;
  }

  // Derive the pre-roll ring capacity (in samples) from the current sample
  // rate and `prerollMs`. Recomputed in start() once the real AudioContext
  // sample rate is known.
  _recomputePreroll() {
    this._prerollMaxSamples =
      this.prerollMs > 0 ? Math.round((this.prerollMs / 1000) * this._sampleRate) : 0;
  }

  // Append a pre-onset frame to the ring buffer and trim the oldest frames so
  // the retained audio never exceeds the pre-roll capacity. No-op when
  // pre-roll is disabled (capacity 0) so the default path stays zero-cost.
  _pushPreroll(frame) {
    if (this._prerollMaxSamples <= 0) return;
    this._prerollChunks.push(frame);
    this._prerollSamples += frame.length;
    while (
      this._prerollChunks.length > 1 &&
      this._prerollSamples - this._prerollChunks[0].length >= this._prerollMaxSamples
    ) {
      this._prerollSamples -= this._prerollChunks.shift().length;
    }
  }

  // Pre-amplify a frame by the configured software gain, returning a new
  // Float32Array. Unity gain (the default) returns the input untouched so the
  // historical path allocates nothing. Mirrors the replay harness `gain` model.
  _applyGain(input) {
    if (this.gain === 1.0) return input;
    const out = new Float32Array(input.length);
    for (let i = 0; i < input.length; i++) out[i] = input[i] * this.gain;
    return out;
  }

  mute() { this._muted = true; }
  unmute() { this._muted = false; }

  setRawRecordingCallback(cb) { this._rawCallback = cb; }

  async start() {
    if (this.active) return;
    this._muted = false;
    this._rawCallback = this._rawCallback || null;

    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: true },
    });

    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    this.audioContext = new AudioCtx();
    await this.audioContext.resume();

    this._sampleRate = this.audioContext.sampleRate || 48000;
    this._recomputePreroll();
    this._prerollChunks = [];
    this._prerollSamples = 0;

    const source = this.audioContext.createMediaStreamSource(this.stream);
    const silentGain = this.audioContext.createGain();
    silentGain.gain.value = 0;

    this.frameCount = 0;
    this.lastRms = 0;

    if (await tryLoadWorklet(this.audioContext, this._workletUrl)) {
      const node = new AudioWorkletNode(this.audioContext, "voice-processor");
      node.port.onmessage = (e) => {
        this._handleFrame(new Float32Array(e.data));
      };
      source.connect(node);
      node.connect(silentGain);
      this._processor = node;
    } else {
      const processor = this.audioContext.createScriptProcessor(4096, 1, 1);
      processor.onaudioprocess = (event) => {
        this._handleFrame(event.inputBuffer.getChannelData(0));
      };
      source.connect(processor);
      processor.connect(silentGain);
      this._processor = processor;
    }

    silentGain.connect(this.audioContext.destination);
    this._source = source;
    this._silentGain = silentGain;

    this.active = true;
    if (this.onStateChange) this.onStateChange("listening");
  }

  _handleFrame(input) {
    if (!this.active) return;
    // Apply the software gain stage first so RMS detection, the raw callback,
    // and the stored segment all see the same amplified signal — equivalent to
    // a gainNode sitting upstream in the audio graph. Unity gain is a no-op.
    input = this._applyGain(input);
    const rms = Math.sqrt(input.reduce((sum, s) => sum + s * s, 0) / input.length);

    this.frameCount++;
    this.lastRms = rms;

    if (this._rawCallback) {
      this._rawCallback(new Float32Array(input), rms);
    }

    if (this._muted) return;

    if (rms > this.silenceThreshold) {
      if (!this.speaking) {
        if (!this._speechCandidate) {
          this._speechCandidate = performance.now();
          this._candidateChunks = [new Float32Array(input)];
        } else {
          this._candidateChunks.push(new Float32Array(input));
          if (performance.now() - this._speechCandidate > 200) {
            this.speaking = true;
            // Prepend the pre-roll ring (pre-onset audio) so the committed
            // segment keeps the quiet soft attack the RMS gate clipped. With
            // pre-roll disabled this drains empty → historical behaviour.
            this.chunks = this._drainPreroll().concat(this._candidateChunks);
            this._candidateChunks = null;
            this._speechCandidate = null;
            this.speechStartedAt = performance.now();
            if (this.onStateChange) this.onStateChange("speaking");
          }
        }
      } else {
        this.lastSpeechAt = performance.now();
        this.chunks.push(new Float32Array(input));
        this._resetSilenceTimer();
      }
    } else {
      // A broken candidate's frames become pre-onset history: fold them into
      // the ring so a later real onset can still reach back across them.
      if (this._candidateChunks) {
        for (const c of this._candidateChunks) this._pushPreroll(c);
      }
      this._speechCandidate = null;
      this._candidateChunks = null;
      if (this.speaking) {
        this.chunks.push(new Float32Array(input));
      } else {
        this._pushPreroll(new Float32Array(input));
      }
    }
  }

  // Return the buffered pre-onset frames (oldest → newest) and reset the ring.
  // Empty array when pre-roll is disabled, preserving the historical segment.
  _drainPreroll() {
    if (this._prerollMaxSamples <= 0) return [];
    const frames = this._prerollChunks;
    this._prerollChunks = [];
    this._prerollSamples = 0;
    return frames;
  }

  _resetSilenceTimer() {
    if (this.silenceTimer) clearTimeout(this.silenceTimer);
    this.silenceTimer = setTimeout(() => this._onSilence(), this.silenceDurationMs);
  }

  _onSilence() {
    if (!this.speaking) return;
    const elapsed = performance.now() - this.speechStartedAt;
    this.speaking = false;

    if (elapsed < this.minSpeechMs || this.chunks.length === 0) {
      if (this.onStateChange) this.onStateChange("listening");
      return;
    }

    const sampleRate = this.audioContext ? this.audioContext.sampleRate : 48000;
    const merged = mergeFloat32Chunks(this.chunks);
    const downsampled = downsampleBuffer(merged, sampleRate, 16000);
    const wav = encodeWav(downsampled, 16000);
    this.chunks = [];

    if (this.onStateChange) this.onStateChange("processing");
    if (this.onSpeechEnd) this.onSpeechEnd(wav);
  }

  async stop() {
    this.active = false;
    this.speaking = false;
    this._prerollChunks = [];
    this._prerollSamples = 0;
    if (this.silenceTimer) clearTimeout(this.silenceTimer);
    if (this._source) this._source.disconnect();
    if (this._processor) {
      if (this._processor.port) this._processor.port.close();
      this._processor.disconnect();
    }
    if (this.stream) {
      for (const track of this.stream.getTracks()) track.stop();
      this.stream = null;
    }
    if (this.audioContext) {
      await this.audioContext.close();
      this.audioContext = null;
    }
  }
}
