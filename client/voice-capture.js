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
  constructor({ onSpeechEnd, onStateChange, workletUrl, silenceThreshold = 0.015, silenceDurationMs = 800, minSpeechMs = 500 } = {}) {
    this.onSpeechEnd = onSpeechEnd;
    this.onStateChange = onStateChange;
    this._workletUrl = workletUrl || null;
    this.silenceThreshold = silenceThreshold;
    this.silenceDurationMs = silenceDurationMs;
    this.minSpeechMs = minSpeechMs;
    this.active = false;
    this.speaking = false;
    this.chunks = [];
    this.speechStartedAt = 0;
    this.lastSpeechAt = 0;
    this.silenceTimer = null;
    this.stream = null;
    this.audioContext = null;
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
            this.chunks = this._candidateChunks;
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
      this._speechCandidate = null;
      this._candidateChunks = null;
      if (this.speaking) {
        this.chunks.push(new Float32Array(input));
      }
    }
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
