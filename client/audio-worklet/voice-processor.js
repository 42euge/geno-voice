class VoiceProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._bufferSize = 4096;
    this._buffer = new Float32Array(this._bufferSize);
    this._writeIndex = 0;

    this.port.onmessage = (e) => {
      const msg = e.data;
      if (msg && msg.type === "config" && typeof msg.bufferSize === "number") {
        this._bufferSize = msg.bufferSize;
        this._buffer = new Float32Array(this._bufferSize);
        this._writeIndex = 0;
      }
    };
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;

    const channel = input[0];
    let offset = 0;

    while (offset < channel.length) {
      const remaining = this._bufferSize - this._writeIndex;
      const toCopy = Math.min(channel.length - offset, remaining);
      this._buffer.set(channel.subarray(offset, offset + toCopy), this._writeIndex);
      this._writeIndex += toCopy;
      offset += toCopy;

      if (this._writeIndex >= this._bufferSize) {
        const transfer = this._buffer.buffer;
        this.port.postMessage(transfer, [transfer]);
        this._buffer = new Float32Array(this._bufferSize);
        this._writeIndex = 0;
      }
    }

    return true;
  }
}

registerProcessor("voice-processor", VoiceProcessor);
