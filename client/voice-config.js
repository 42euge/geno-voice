const DEFAULT_SERVER_URL = "http://127.0.0.1:5111";
const VOICE_PREFS_KEY = "mind-render.voice-preferences";
const LOCAL_HOSTS = new Set(["127.0.0.1", "localhost"]);

export const DEFAULT_VOICE_PREFERENCES = Object.freeze({
  serverUrl: DEFAULT_SERVER_URL,
  ttsEnabled: false,
});

export function normalizeLocalVoiceUrl(
  input,
  fallback = DEFAULT_SERVER_URL,
  { strict = false } = {}
) {
  const candidate = (input || fallback || DEFAULT_SERVER_URL).trim();
  try {
    const parsed = new URL(candidate);
    if (parsed.protocol !== "http:") {
      throw new Error("Voice server URL must use http://");
    }
    if (!LOCAL_HOSTS.has(parsed.hostname)) {
      throw new Error(
        "Voice server URL must point to localhost or 127.0.0.1"
      );
    }
    parsed.pathname = "";
    parsed.search = "";
    parsed.hash = "";
    return parsed.toString().replace(/\/$/, "");
  } catch (error) {
    if (strict) throw error;
    if (candidate === fallback) return DEFAULT_SERVER_URL;
    return normalizeLocalVoiceUrl(fallback, DEFAULT_SERVER_URL, { strict: true });
  }
}

export function loadVoicePreferences(defaultServerUrl = DEFAULT_SERVER_URL) {
  let stored = null;
  try {
    stored = JSON.parse(localStorage.getItem(VOICE_PREFS_KEY) || "null");
  } catch (error) {
    stored = null;
  }

  return {
    ...DEFAULT_VOICE_PREFERENCES,
    ...stored,
    serverUrl: normalizeLocalVoiceUrl(
      stored && stored.serverUrl ? stored.serverUrl : defaultServerUrl,
      defaultServerUrl
    ),
    ttsEnabled:
      stored && typeof stored.ttsEnabled === "boolean"
        ? stored.ttsEnabled
        : DEFAULT_VOICE_PREFERENCES.ttsEnabled,
  };
}

export function saveVoicePreferences(preferences) {
  localStorage.setItem(VOICE_PREFS_KEY, JSON.stringify(preferences));
}

export function toVoiceWebSocketUrl(serverUrl) {
  const parsed = new URL(normalizeLocalVoiceUrl(serverUrl));
  parsed.protocol = parsed.protocol === "https:" ? "wss:" : "ws:";
  parsed.pathname = "/tts/stream";
  return parsed.toString();
}

export function createVoiceConfigPatch({ sttEngine, ttsVoice, ttsSpeed }) {
  return {
    stt: {
      engine: sttEngine,
    },
    tts: {
      voice: ttsVoice,
      speed: Number(ttsSpeed),
    },
  };
}
