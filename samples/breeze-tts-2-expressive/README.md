# Breeze TTS 2 expressive samples

This set deliberately pushes the live `geno-voice` Breeze TTS 2 endpoint
across emotion, pacing, vocal events, character design, bilingual delivery,
and difficult prosody. These are deterministic first-pass generations (seed
42), not cherry-picked takes.

The samples are mono, 24 kHz, signed 16-bit PCM WAV files. See
[`manifest.json`](manifest.json) for the exact text, direction, intended
capability, measured duration, and SHA-256 digest of every clip.

## Listening order

Start with `01` through `08` for emotional range and voice-agent behavior,
then compare `09` and `10` for pace control. Clips `11` through `16` stress
inline vocal events, voice design, Mandarin, code-switching, punctuation,
and near-sung delivery.

On macOS, play one clip with:

```bash
afplay samples/breeze-tts-2-expressive/01_intimate_whisper_en.wav
```

Regenerate the full set from the repository root with:

```bash
python samples/breeze-tts-2-expressive/generate.py \
  --endpoint ws://127.0.0.1:8765/v1/tts/stream \
  --overwrite
```

The generator uses one persistent WebSocket connection and submits clips
serially because the official Breeze runtime supports a single active
synthesis request.

## License

Breeze TTS 2 weights and self-hosted outputs are governed by the BreezeBlue
Research and Non-Commercial License. These samples are for research and
evaluation, not commercial use.
