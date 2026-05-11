#!/usr/bin/env python3
"""
Batch-process multiple podcast episodes through the pipeline.
Produces per-episode results + combined training dataset.
"""

import io, json, sys, time, wave
from pathlib import Path
import requests

VOICE_URL = "http://127.0.0.1:5111"
DATA = Path(__file__).parent.parent / "test-data"
SYSTEM = (
    "You are a reflective companion using motivational interviewing. "
    "Paraphrase what you hear, use open questions, affirmations, and reflections. "
    "Do not diagnose, prescribe, or give advice. You are not a therapist."
)

def process_episode(wav_path, skip_secs=30):
    name = wav_path.stem
    print(f"\n{'='*60}\n  {name}\n{'='*60}")

    with wave.open(str(wav_path), "rb") as w:
        sr = w.getframerate(); ch = w.getnchannels(); sw = w.getsampwidth()
        total = w.getnframes() / sr
        w.setpos(int(skip_secs * sr))
        frames = w.readframes(w.getnframes() - int(skip_secs * sr))

    bps = sr * ch * sw
    cb = int(8 * bps)
    chunks = []
    for i in range(0, len(frames), cb):
        c = frames[i:i+cb]
        if len(c) < bps: break
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(ch); w.setsampwidth(sw); w.setframerate(sr)
            w.writeframes(c)
        chunks.append(buf.getvalue())

    print(f"  {len(chunks)} chunks ({total:.0f}s total)")
    results = []
    for i, wav in enumerate(chunks):
        try:
            resp = requests.post(f"{VOICE_URL}/stt/transcribe", data=wav,
                headers={"Content-Type": "audio/wav"}, timeout=120)
            d = resp.json()
            text = d.get("text", "")
            if text:
                requests.post(f"{VOICE_URL}/notes/process", json={"text": text}, timeout=5).close()
            results.append({"chunk": i+1, "text": text, "trigger": d.get("trigger"),
                "filtered": d.get("filtered", False)})
        except Exception as e:
            results.append({"chunk": i+1, "error": str(e)})
        if (i+1) % 20 == 0:
            print(f"  {i+1}/{len(chunks)}")

    out = DATA / f"results-{name}.json"
    out.write_text(json.dumps(results, indent=2))
    triggers = sum(1 for r in results if r.get("trigger"))
    errors = sum(1 for r in results if r.get("error"))
    print(f"  Done: {len(results)} chunks, {triggers} triggers, {errors} errors → {out.name}")
    return name, results


def build_dataset(all_results):
    dataset = []
    for name, results in all_results:
        context = []
        silent_ctx = []
        for r in results:
            text = r.get("text", "").strip()
            if not text or r.get("filtered") or r.get("error"): continue
            trigger = r.get("trigger")
            context.append({"role": "user", "content": text})
            if trigger:
                dataset.append({"messages": [{"role": "system", "content": SYSTEM}, *context[-4:]],
                    "metadata": {"trigger": trigger, "source": name}})
                silent_ctx = []
            else:
                silent_ctx.append(text)
                if len(silent_ctx) >= 3:
                    dataset.append({"messages": [
                        {"role": "system", "content": SYSTEM + "\n\nThe user is speaking. Stay silent and listen unless directly addressed."},
                        *[{"role": "user", "content": t} for t in silent_ctx[-3:]],
                        {"role": "assistant", "content": "[listening]"}],
                        "metadata": {"action": "stay_silent", "source": name}})
                    silent_ctx = []
            if len(context) > 6: context = context[-4:]
    return dataset


def main():
    episodes = sorted(DATA.glob("esther-perel-ep*.wav"))
    if not episodes:
        print("No episodes found in test-data/")
        return

    print(f"Found {len(episodes)} episodes")
    all_results = []

    # Load existing results if available
    for ep in episodes:
        cached = DATA / f"results-{ep.stem}.json"
        if cached.exists():
            results = json.loads(cached.read_text())
            print(f"  {ep.stem}: cached ({len(results)} chunks)")
            all_results.append((ep.stem, results))
        else:
            name, results = process_episode(ep)
            all_results.append((name, results))

    # Also include ep1 and ep2 if available
    for f in ["podcast-test-results.json", "podcast-test-results-ep2.json"]:
        p = DATA / f
        if p.exists():
            results = json.loads(p.read_text())
            name = f.replace("podcast-test-results", "ep-main").replace(".json", "")
            all_results.append((name, results))
            print(f"  {name}: {len(results)} chunks")

    dataset = build_dataset(all_results)
    out = DATA / "training-dataset.jsonl"
    out.write_text("\n".join(json.dumps(d) for d in dataset))

    respond = sum(1 for d in dataset if d.get("metadata", {}).get("trigger"))
    silent = sum(1 for d in dataset if d.get("metadata", {}).get("action") == "stay_silent")
    print(f"\nCombined dataset: {len(dataset)} examples ({respond} respond, {silent} silent)")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
