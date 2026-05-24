#!/usr/bin/env python3
"""
Build training dataset from podcast test results + session notes.

Combines STT transcriptions, trigger detections, and session context
into a structured JSONL dataset for fine-tuning.

Usage:
    .venv/bin/python examples/build_dataset.py
"""

import json
from pathlib import Path

RESULTS = Path(__file__).parent.parent / "test-data" / "podcast-test-results.json"
OUT = Path(__file__).parent.parent / "test-data" / "training-dataset.jsonl"

SYSTEM_PROMPT = (
    "You are a reflective companion using motivational interviewing. "
    "Paraphrase what you hear, use open questions, affirmations, and reflections. "
    "Do not diagnose, prescribe, or give advice. You are not a therapist."
)


def main():
    if not RESULTS.exists():
        print(f"No results at {RESULTS}")
        print("Run: .venv/bin/python examples/podcast_test.py --start 60 --duration 2700")
        return

    results = json.loads(RESULTS.read_text())
    print(f"Loaded {len(results)} chunks")

    # Build conversation windows — each training example is
    # context (previous 2-3 utterances) + the current utterance
    dataset = []
    context = []

    for r in results:
        text = r.get("text", "").strip()
        if not text or r.get("filtered"):
            continue

        trigger = r.get("trigger")
        context.append({"role": "user", "content": text})

        # When a trigger is detected, the system should respond
        # Create a training example where the model learns WHEN to respond
        if trigger:
            example = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    *context[-4:],  # last 4 turns as context
                ],
                "metadata": {
                    "trigger": trigger,
                    "time_offset": r.get("time_offset"),
                    "source": "esther-perel-where-should-we-begin",
                },
            }
            dataset.append(example)

        # Keep context window manageable
        if len(context) > 6:
            context = context[-4:]

    # Also create examples where the system should stay silent
    silent_context = []
    for r in results:
        text = r.get("text", "").strip()
        if not text or r.get("filtered") or r.get("trigger"):
            continue
        silent_context.append(text)
        if len(silent_context) >= 3:
            # 3 consecutive non-trigger utterances = system should stay silent
            dataset.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT + "\n\nThe user is speaking. Stay silent and listen unless directly addressed."},
                    *[{"role": "user", "content": t} for t in silent_context[-3:]],
                    {"role": "assistant", "content": "[listening]"},
                ],
                "metadata": {
                    "trigger": None,
                    "action": "stay_silent",
                    "source": "esther-perel-where-should-we-begin",
                },
            })
            silent_context = []

    with open(OUT, "w") as f:
        for ex in dataset:
            f.write(json.dumps(ex) + "\n")

    respond = sum(1 for d in dataset if d.get("metadata", {}).get("trigger"))
    silent = sum(1 for d in dataset if d.get("metadata", {}).get("action") == "stay_silent")
    print(f"\nDataset: {len(dataset)} examples")
    print(f"  Respond: {respond}")
    print(f"  Silent: {silent}")
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
