import { normalizeLocalVoiceUrl, toVoiceWebSocketUrl } from "./voice-config.js";

const SENTENCE_PUNCTUATION = new Set([".", "!", "?", "\n"]);
const TRAILING_SENTENCE_CHARS = new Set(['"', "'", ")", "]", "}", " ", "\n"]);

export class StreamingAudioPlayer {
  constructor({ playChunk, stopPlayback } = {}) {
    this.currentToken = 0;
    this.activeItem = null;
    this.queue = [];
    this.playChunk = playChunk || null;
    this.stopPlayback = stopPlayback || null;
  }

  async ensureContext() {
    if (typeof Audio === "undefined") {
      throw new Error("Audio playback is not available in this environment.");
    }
    return true;
  }

  async begin() {
    this.currentToken += 1;
    this.stopAll();
    await this.ensureContext();
    return this.currentToken;
  }

  async enqueueChunk(chunk, token) {
    await this.ensureContext();
    if (token !== this.currentToken) return;

    const audioData =
      chunk instanceof ArrayBuffer
        ? chunk.slice(0)
        : chunk.buffer.slice(chunk.byteOffset, chunk.byteOffset + chunk.byteLength);

    if (this.playChunk) {
      return this.playChunk(new Uint8Array(audioData));
    }

    const blob = new Blob([audioData], { type: "audio/wav" });
    const url = URL.createObjectURL(blob);

    return new Promise((resolve, reject) => {
      this.queue.push({ token, url, resolve, reject });
      this.playNext();
    });
  }

  cancel() {
    this.currentToken += 1;
    this.stopAll();
  }

  playNext() {
    if (this.activeItem) return;
    if (this.queue.length === 0) return;

    const item = this.queue.shift();
    if (!item || item.token !== this.currentToken) {
      if (item) {
        URL.revokeObjectURL(item.url);
        item.resolve();
      }
      this.playNext();
      return;
    }

    const audio = new Audio(item.url);
    audio.preload = "auto";
    this.activeItem = { audio, item };

    const finalize = (callback) => {
      audio.onended = null;
      audio.onerror = null;
      audio.onpause = null;
      URL.revokeObjectURL(item.url);
      if (this.activeItem && this.activeItem.audio === audio) {
        this.activeItem = null;
      }
      callback();
      this.playNext();
    };

    audio.onended = () => {
      finalize(() => item.resolve());
    };

    audio.onerror = () => {
      finalize(() => item.reject(new Error("Audio playback failed.")));
    };

    const playAttempt = audio.play();
    if (playAttempt && typeof playAttempt.catch === "function") {
      playAttempt.catch((error) => {
        finalize(() =>
          item.reject(
            error instanceof Error
              ? error
              : new Error("Audio playback failed.")
          )
        );
      });
    }
  }

  stopAll() {
    if (this.stopPlayback) {
      this.stopPlayback();
    }

    if (this.activeItem) {
      try {
        this.activeItem.audio.pause();
        this.activeItem.audio.removeAttribute("src");
        this.activeItem.audio.load();
      } catch (error) {
        // ignore teardown failures during cancel
      }
      URL.revokeObjectURL(this.activeItem.item.url);
      this.activeItem.item.resolve();
      this.activeItem = null;
    }

    while (this.queue.length > 0) {
      const item = this.queue.shift();
      URL.revokeObjectURL(item.url);
      item.resolve();
    }
  }
}

export class VoiceStreamClient {
  constructor(player, { onError } = {}) {
    this.player = player;
    this.onError = onError;
    this.socket = null;
    this.token = 0;
    this.ready = null;
  }

  async start({ serverUrl }) {
    this.stop();
    this.token = await this.player.begin();

    const wsUrl = toVoiceWebSocketUrl(serverUrl);
    this.socket = new WebSocket(wsUrl);
    this.socket.binaryType = "arraybuffer";

    const token = this.token;
    this.ready = new Promise((resolve, reject) => {
      this.socket.addEventListener(
        "open",
        () => {
          resolve();
        },
        { once: true }
      );
      this.socket.addEventListener(
        "error",
        () => {
          reject(new Error("Voice streaming connection failed."));
        },
        { once: true }
      );
    });

    this.socket.addEventListener("message", async (event) => {
      if (token !== this.token) return;
      try {
        await this.player.enqueueChunk(event.data, token);
      } catch (error) {
        if (this.onError) this.onError(error);
      }
    });

    this.socket.addEventListener("close", () => {
      if (this.socket && token === this.token) {
        this.socket = null;
        this.ready = null;
      }
    });

    return this.ready;
  }

  async speak(payload) {
    if (!this.socket || !this.ready) {
      throw new Error("Voice streaming is not ready.");
    }
    await this.ready;
    this.socket.send(JSON.stringify(payload));
  }

  stop() {
    if (this.socket) {
      this.socket.close();
      this.socket = null;
      this.ready = null;
    }
    this.player.cancel();
  }
}

function stripSpeechHiddenBlocks(text) {
  let result = "";
  let cursor = 0;

  while (cursor < text.length) {
    const blockStart = text.indexOf("```", cursor);
    if (blockStart === -1) {
      result += text.slice(cursor);
      break;
    }

    result += text.slice(cursor, blockStart);
    const blockEnd = text.indexOf("```", blockStart + 3);
    if (blockEnd === -1) break;
    cursor = blockEnd + 3;
  }

  return result;
}

function collapseWhitespace(text) {
  return text.replace(/\s+/g, " ").trim();
}

function isSentenceBoundary(text, index) {
  const char = text[index];
  if (!SENTENCE_PUNCTUATION.has(char)) return false;
  if (char === "\n") return true;
  const next = text[index + 1];
  if (!next) return true;
  if (/\s/.test(next)) return true;
  if (TRAILING_SENTENCE_CHARS.has(next)) {
    const afterTrailing = text[index + 2];
    return !afterTrailing || /\s/.test(afterTrailing);
  }
  return false;
}

export function pullCompleteSpeechSegments(text, cursor, { final = false } = {}) {
  const speakable = stripSpeechHiddenBlocks(text);
  const segments = [];
  let start = cursor;

  for (let i = cursor; i < speakable.length; i++) {
    if (!isSentenceBoundary(speakable, i)) continue;

    let end = i + 1;
    while (end < speakable.length && TRAILING_SENTENCE_CHARS.has(speakable[end])) {
      end += 1;
    }

    const candidate = collapseWhitespace(speakable.slice(start, end));
    if (candidate) segments.push(candidate);
    start = end;
  }

  if (final && start < speakable.length) {
    const tail = collapseWhitespace(speakable.slice(start));
    if (tail) segments.push(tail);
    start = speakable.length;
  }

  return {
    cursor: start,
    segments,
    speakableText: speakable,
  };
}
