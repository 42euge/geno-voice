# GenoVoice overlay prototype

THROWAWAY UI PROTOTYPE — three variants of the transient GenoVoice surface,
switchable via `?variant=`, on the standalone prototype route.

Question: which structure makes a spoken one-off question and its answer easiest
to understand without turning GenoVoice into another chat window?

Run it from the repository root:

```bash
./scripts/run_quick_agent_ui_prototype.sh
```

Then open <http://127.0.0.1:4173/?variant=A>. Use the floating arrows or the
left/right arrow keys to compare:

- A — Capsule: the Superwhisper-like compact recording bar expands in place.
- B — Command palette: the question and answer replace each other in a larger
  Spotlight-like surface.
- C — Side glance: a narrow answer card stays out of the current app's center.

Click the main surface to replay its listening → thinking → answer states.
There are no network requests or persisted settings.
