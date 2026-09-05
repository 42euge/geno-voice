# Full-duplex voice papers, explained simply

_Research snapshot: 2026-08-27. This guide covers every paper in the
[offline archive](papers/full-duplex/README.md)._

## The whole field in one picture

A conventional voice assistant repeats this sequence:

```text
listen -> transcribe -> think -> speak -> listen again
```

A full-duplex system keeps listening while it speaks. That creates a harder
problem than speech recognition: the system must decide whether new sound is
its own echo, noise, a backchannel such as “mm-hmm,” speech to somebody else,
or a real interruption. The recent literature attacks that problem in four
ways:

1. **Native duplex models** learn two synchronized audio streams jointly.
2. **Pluggable controllers** put a fast interaction policy around an existing
   ASR/LLM/TTS system.
3. **Turn models** predict whether speech is starting, continuing, or ending.
4. **Benchmarks** test overlap, interruption, disfluency, and tool use.

The useful mathematical abstraction is two concurrent signals: user activity
$u_t$ and assistant activity $a_t$. Instead of predicting only the next word,
a duplex system continually estimates both conversation and control:

$$
P(u_{t+1:t+H}, a_{t+1:t+H}, c_t \mid u_{\le t}, a_{\le t}),
$$

where $c_t$ is an action such as listen, duck, resume, interrupt, or speak.

## Existing surveys

### WavChat: A Survey of Spoken Dialogue Models (2024)

[Local PDF](papers/full-duplex/2411.13577v2.pdf) ·
[Primary source](https://arxiv.org/abs/2411.13577)

- **In simple terms:** This is the broad map of modern spoken dialogue, not
  only duplex. It covers cascaded and end-to-end systems, speech
  representations, training, streaming, datasets, and evaluation.
- **What it contributes:** A chronological catalog and a vocabulary for the
  components surrounding duplex interaction.
- **What to borrow:** Use it for background on codecs, speech encoders, and
  training paradigms. Its duplex section is orientation rather than an
  implementation recipe.
- **Caution:** Its scope is so broad that interaction control, AEC, and safe
  tool execution receive less depth than geno-voice needs.

### From Turn-Taking to Synchronous Dialogue (2025)

[Local PDF](papers/full-duplex/2509.14515v1.pdf) ·
[Primary source](https://arxiv.org/abs/2509.14515)

- **In simple terms:** The paper separates systems that bolt synchronization
  onto modules from systems that learn synchronization inside one model.
- **What it contributes:** The useful split between **engineered
  synchronization** and **learned synchronization**, plus four evaluation
  pillars: timing, behavioral arbitration, meaning, and acoustic quality.
- **What to borrow:** Evaluate those pillars separately. A system can stop
  quickly yet misunderstand the interruption, or sound good while managing
  turns badly.
- **Caution:** Its model-centric taxonomy does not fully specify production
  details such as echo references, playback cursors, or irreversible tools.

### A Survey of Full-Duplex Spoken Dialogue Systems (2026)

[Local PDF](papers/full-duplex/2606.19453v1.pdf) ·
[Primary source](https://arxiv.org/abs/2606.19453)

- **In simple terms:** “Full duplex” is too vague, so this paper describes
  where the decision is made, what kind of overlap occurred, and what response
  is required.
- **What it contributes:** An L0--L3 architecture hierarchy, a
  $T\times I\times R$ interaction ontology (timing, intent, response), and a
  five-state machine: idle, listen, speak, wait, and dual.
- **What to borrow:** Describe a system by its supported interaction cells,
  not by a single full-duplex label. Use its state vocabulary when comparing
  implementations.
- **Caution:** Its L3 shared-latent architecture remains a research hypothesis,
  and parts of its audit depend on systems' self-reported behavior.

## Foundations and turn-taking signals

### Generative Spoken Dialogue Language Modeling, dGSLM (2022/2023)

[Local PDF](papers/full-duplex/2203.16502v2.pdf) ·
[Primary source](https://arxiv.org/abs/2203.16502)

- **In simple terms:** Train on both sides of real conversations and the model
  can learn timing, overlap, laughter, and turn exchange without first turning
  everything into text.
- **How it works:** Each speaker becomes a stream of discrete speech units and
  the model learns their joint distribution:

  $$P(X^A, X^B)=\prod_t P(x^A_t,x^B_t\mid X^A_{<t},X^B_{<t}).$$

- **Why it matters:** It established that two synchronized timelines are the
  natural representation for duplex speech.
- **Caution:** It is a foundational generative model, not a deployable coding
  agent or a drop-in turn controller.

### Voice Activity Projection, VAP (2022)

[Local PDF](papers/full-duplex/2205.09812v1.pdf) ·
[Primary source](https://arxiv.org/abs/2205.09812)

- **In simple terms:** Ordinary VAD asks “who is speaking now?” VAP asks “who
  is likely to speak during the next few moments?”
- **How it works:** From both speakers' recent audio, it predicts future binary
  activity over a horizon $H$:

  $$\hat{Y}_{t:t+H}=f_\theta(X^A_{\le t},X^B_{\le t}).$$

- **Why it matters:** SHIFT/HOLD, backchannel opportunity, and turn-yield
  predictions emerge from one acoustic model.
- **Caution:** The standard model expects separated speaker channels; a laptop
  microphone containing assistant echo is not the same input distribution.

### Real-time and Continuous Turn-taking Prediction Using VAP (2024)

[Local PDF](papers/full-duplex/2401.04868v1.pdf) ·
[Primary source](https://arxiv.org/abs/2401.04868)

- **In simple terms:** This paper asks whether VAP can run continuously in a
  live system instead of only on recorded conversations.
- **What it contributes:** A real-time CPU implementation and analysis of
  continuous turn-taking decisions.
- **What to borrow:** VAP is practical enough to benchmark as an optional
  acoustic projection signal after echo cancellation.
- **Caution:** Real time does not mean correct under room echo, competing
  speakers, or coding-specific language.

### “Yeah, Un, Oh”: Backchannel Prediction with VAP (2024)

[Local PDF](papers/full-duplex/2410.15929v2.pdf) ·
[Primary source](https://arxiv.org/abs/2410.15929)

- **In simple terms:** A listener's “yeah” should support the speaker, not steal
  the turn. This work fine-tunes VAP to predict backchannel timing and type.
- **What it contributes:** Continuous prediction instead of deciding only at
  hand-picked silence boundaries.
- **What to borrow:** Backchannel is a separate interaction intent, not a short
  interruption detected by duration alone.
- **Caution:** Backchannel norms vary by speaker, language, and task.

### Applying General Turn-taking Models to Conversational HRI (2025)

[Local PDF](papers/full-duplex/2501.08946v1.pdf) ·
[Primary source](https://arxiv.org/abs/2501.08946)

- **In simple terms:** Audio prosody and transcript meaning make different
  mistakes, so the paper combines acoustic VAP with textual TurnGPT.
- **What it contributes:** Evidence that acoustic and linguistic turn signals
  are complementary in a live human--robot setting.
- **What to borrow:** Fuse probabilities or features; do not replace every
  signal with one threshold.
- **Caution:** Fusion quality depends on streaming transcripts arriving soon
  enough to affect the decision.

## Native models and pluggable controllers

### SyncLLM (2024)

[Local PDF](papers/full-duplex/2409.15594v1.pdf) ·
[Primary source](https://arxiv.org/abs/2409.15594)

- **In simple terms:** Make an LLM experience conversation as a wall-clock
  sequence of short synchronized audio chunks rather than alternating turns.
- **What it contributes:** Explicit temporal synchronization and prediction of
  the user chunk that has not fully arrived when the assistant must generate.
- **What to borrow:** Timestamp every input, decision, and rendered sample;
  include network delay in the state.
- **Caution:** The project releases examples but no runnable model weights.

### Moshi (2024)

[Local PDF](papers/full-duplex/2410.00037v2.pdf) ·
[Primary source](https://arxiv.org/abs/2410.00037)

- **In simple terms:** One 7B model jointly follows user audio, assistant
  speech, and an assistant “inner monologue” text stream in real time.
- **How it works:** Delayed multi-stream modeling makes several codebooks
  autoregressive without serializing every audio token onto one slow timeline.
- **Why it matters:** It is the clearest obtainable latency and behavior
  baseline for a native duplex model, including an MLX implementation.
- **Caution:** Its speech policy and reasoning model are coupled, so it cannot
  simply become the audio layer for a Blue/custom coding endpoint.

### FlexDuo (2025)

[Local PDF](papers/full-duplex/2502.13472v2.pdf) ·
[Primary source](https://arxiv.org/abs/2502.13472)

- **In simple terms:** Put a learned traffic controller around a half-duplex
  assistant. Every 120 ms it decides whether to keep speaking, listening, or
  idling, or to transition between them.
- **How it works:** A policy maps recent audio and state to seven actions:
  $a_t\sim\pi_\theta(a\mid o_{\le t},s_t)$.
- **Why it matters:** Its explicit idle state prevents noise and irrelevant
  speech from contaminating the conversation.
- **Caution:** Its 7B controller is too large for our first local controller,
  and code/weights are not released.

### LLM-Enhanced Dialogue Management for Full-Duplex Systems (2025)

[Local PDF](papers/full-duplex/2502.14145v3.pdf) ·
[Primary source](https://arxiv.org/abs/2502.14145)

- **In simple terms:** A small semantic model decides whether the assistant
  should keep listening, start speaking, start listening, or keep speaking.
- **What it contributes:** A 0.5B dialogue manager placed after AEC, VAD, and
  ASR but before the larger dialogue engine.
- **What to borrow:** Keep interaction control small, local, and separate from
  the expensive reasoning model.
- **Caution:** The released artifact is data-generation code, not a ready
  controller checkpoint.

### SALM-Duplex (2025)

[Local PDF](papers/full-duplex/2505.15670v4.pdf) ·
[Primary source](https://arxiv.org/abs/2505.15670)

- **In simple terms:** A relatively small language model can learn duplex
  behavior when it continuously receives user speech while generating
  assistant text and audio.
- **What it contributes:** An 80 ms streaming user encoder and separate text
  and codec-audio output channels without requiring speech--text pretraining.
- **What to borrow:** Separate perception timing from voice generation timing.
- **Caution:** The reported 0.64 s yield delay is conservative, and reproducing
  training still requires substantial GPU resources and data.

### FireRedChat (2025)

[Local PDF](papers/full-duplex/2509.06502v1.pdf) ·
[Primary source](https://arxiv.org/abs/2509.06502)

- **In simple terms:** This is the most direct open engineering reference: a
  personalized VAD identifies the intended user, a semantic model detects turn
  completion, and a replaceable dialogue manager handles the response.
- **What it contributes:** Released pVAD and turn-detector models plus a
  self-hosted cascaded implementation.
- **What to borrow:** Benchmark its pVAD and end-of-turn components before
  training geno-voice equivalents.
- **Caution:** Its full deployment stack is much heavier than importing two
  focused inference components.

### F-Actor (2026)

[Local PDF](papers/full-duplex/2601.11329v3.pdf) ·
[Primary source](https://arxiv.org/abs/2601.11329)

- **In simple terms:** Tell the model how conversational it should be, including
  how often it backchannels or interrupts, instead of baking one personality
  into the weights.
- **What it contributes:** Open model/training code and instruction control
  over voice, topic, dialogue initiation, backchannels, and interruption.
- **What to borrow:** Make turn-taking policy a session-level configuration.
- **Caution:** It is a native speech model benchmark, not an external-agent
  controller.

### DuplexSLA (2026)

[Local PDF](papers/full-duplex/2605.20755v2.pdf) ·
[Primary source](https://arxiv.org/abs/2605.20755)

- **In simple terms:** Speech, language, and actions need separate synchronized
  lanes. A tool call should not be smuggled through spoken text.
- **What it contributes:** A rate-limited textual action channel sharing a
  160 ms clock with user and assistant audio.
- **What to borrow:** Give interaction decisions and tool events typed,
  timestamped records.
- **Caution:** At the research snapshot, inference code and checkpoints were
  still announced rather than released.

### IRAF (2026)

[Local PDF](papers/full-duplex/2606.06559v1.pdf) ·
[Primary source](https://arxiv.org/abs/2606.06559)

- **In simple terms:** Do not let every voice in the room influence the agent
  equally. Compare incoming audio with the enrolled user's voice and attenuate
  unreliable frames.
- **How it works:** A causal network produces a frame gate

  $$g_t=2\,\sigma(f_\psi(s,X_{\le t})),\qquad Z_t=g_tX_t+Y_t,$$

  where $s$ is a target-speaker embedding, $X_t$ user audio, and $Y_t$ the
  agent-side stream.
- **What to borrow:** Expose target-speaker confidence separately from generic
  speech probability.
- **Caution:** It improves an end-to-end model; a modular implementation still
  needs to calibrate speaker verification under device and room changes.

### DuplexOmni (2026)

[Local PDF](papers/full-duplex/2606.09186v1.pdf) ·
[Primary source](https://arxiv.org/abs/2606.09186)

- **In simple terms:** Use one fast layer to manage the conversation and a
  slower asynchronous layer to think. The fast layer can abandon obsolete
  thinking and incorporate results when they arrive.
- **What it contributes:** The interaction/thinking split closest to the
  geno-voice plus Blue/custom-endpoint architecture.
- **What to borrow:** Never block the audio controller on tools or deep
  reasoning; identify and supersede stale requests.
- **Caution:** It is paper-only and trained at a scale far beyond this project.

### JoyAI-Talker (2026)

[Local PDF](papers/full-duplex/2608.01119v1.pdf) ·
[Primary source](https://arxiv.org/abs/2608.01119)

- **In simple terms:** Separate deciding what to say from rendering how it
  sounds, while a plug-in duplex gate controls interaction.
- **What it contributes:** Natural-language controls for emotion, rate, vocal
  effort, laughter, and sighs in the Talker.
- **What to borrow:** Put a speech-plan interface between agent text and TTS so
  expressivity can improve without rewriting turn control.
- **Caution:** The reported model is a very large MoE and was paper-only at the
  research snapshot.

## Endpointing and interaction arbitration

### SpeculativeETD (2025)

[Local PDF](papers/full-duplex/2503.23439v2.pdf) ·
[Primary source](https://arxiv.org/abs/2503.23439)

- **In simple terms:** Use a tiny model constantly, and wake a larger model only
  when silence might mean the user is finished rather than merely thinking.
- **What it contributes:** A GRU on 100 ms chunks plus a 94M Wav2Vec2 model
  invoked after 200 ms of silence, reducing expensive inference by more than
  $10\times$ in the paper.
- **What to borrow:** Cascades should increase confidence and compute, not force
  the conversation through sequential turns.
- **Caution:** Artifact licensing is not sufficiently explicit for blind
  vendoring.

### FastTurn (2026)

[Local PDF](papers/full-duplex/2604.01897v6.pdf) ·
[Primary source](https://arxiv.org/abs/2604.01897)

- **In simple terms:** Combine partial transcript meaning with audio timing and
  prosody, then classify complete, incomplete, backchannel, or wait.
- **What it contributes:** Streaming CTC text, Conformer audio features, and a
  Qwen3-0.6B semantic branch fused into a turn head.
- **What to borrow:** Its label set and test data are a strong design target for
  a smaller geno-voice arbiter.
- **Caution:** The roughly 700M-parameter system and unreleased weights make it
  a reference rather than a dependency.

### Endpoint Anticipation (2026)

[Local PDF](papers/full-duplex/2606.13450v1.pdf) ·
[Primary source](https://arxiv.org/abs/2606.13450)

- **In simple terms:** Predict that the user is about to finish, privately
  generate a short answer, and play it only if the endpoint really occurs.
- **What it contributes:** Forecasts up to 2.56 s ahead and reports 505 ms
  average latency reduction for 28.4% extra speculative computation.
- **What to borrow:** Speculative Blue/TTS work can hide a cascaded pipeline's
  latency when output remains inaudible and side-effect-free until commit.
- **Caution:** Wrong forecasts waste compute; speculative tool mutations are
  unsafe and must remain prohibited.

### Next-Turn (2026)

[Local PDF](papers/full-duplex/2606.18094v1.pdf) ·
[Primary source](https://arxiv.org/abs/2606.18094)

- **In simple terms:** Instead of asking only “is the turn over?”, predict how
  long until speech resumes.
- **How it works:** Train a duration head for
  $\Delta_t=t_{\text{next onset}}-t$ and fuse it with binary endpoint
  probability.
- **What it contributes:** A target derived from timestamps without manual
  semantic labels; the paper reports a 25.9-point absolute gain in endpoint
  accuracy within 320 ms over its strongest baseline.
- **What to borrow:** Use time-to-next-onset as an additional label for the
  Superwhisper evaluation corpus.

## Benchmarks

### Full-Duplex-Bench v1.5 (2025)

[Local PDF](papers/full-duplex/2507.23159v4.pdf) ·
[Primary source](https://arxiv.org/abs/2507.23159)

- **In simple terms:** Play controlled overlaps and check whether the model
  should respond or resume.
- **What it contributes:** Cases for genuine interruption, backchannel,
  other-addressed speech, and background speech, plus stop/response latency.
- **What to borrow:** Adopt `RESPOND`/`RESUME` labels and report false stops,
  not only how quickly the system stops.
- **Caution:** Short controlled cases do not cover long-horizon tool state.

### Full-Duplex-Bench v2 (2025)

[Local PDF](papers/full-duplex/2510.07838v2.pdf) ·
[Primary source](https://arxiv.org/abs/2510.07838)

- **In simple terms:** Extend duplex evaluation from isolated events to a live
  multi-turn exchange driven by an automated examiner.
- **What it contributes:** A framework for testing whether behavior remains
  coherent across consecutive interruptions and resumptions.
- **What to borrow:** Use automated multi-turn adversarial playback after the
  deterministic two-channel unit scenarios pass.
- **Caution:** An automatic examiner is only as reliable as its prompts and
  audio timing implementation.

### Full-Duplex-Bench v3 (2026)

[Local PDF](papers/full-duplex/2604.04847v1.pdf) ·
[Primary source](https://arxiv.org/abs/2604.04847)

- **In simple terms:** Test real human disfluency while the voice agent must
  execute multi-step tools, not merely answer a spoken question.
- **What it contributes:** Real audio with five disfluency categories, chained
  mock APIs in four domains, and joint accuracy/latency/turn-taking metrics.
- **What to borrow:** Test self-correction, changed arguments, and stale tool
  plans explicitly—the closest benchmark to a coding agent.
- **Caution:** Aggregate pass rates hide which layer failed; preserve
  controller, model, and tool traces.

### HumDial-FDBench (2026)

[Local PDF](papers/full-duplex/2604.21406v2.pdf) ·
[Primary source](https://arxiv.org/abs/2604.21406)

- **In simple terms:** Evaluate with two-channel conversations recorded by real
  humans, including overlap, interruption, and feedback, rather than relying
  only on synthetic mixtures.
- **What it contributes:** A challenge dataset, behavioral evaluation, and a
  public leaderboard.
- **What to borrow:** Add real dual-channel data as an external-validity check
  after tests built from the Superwhisper corpus.
- **Caution:** Review dataset terms before copying the audio into this
  repository.

## What the papers collectively recommend for geno-voice

The papers do not point to a single winning model. They point to a separation
of responsibilities:

$$
\underbrace{\hat{u}_t=\operatorname{AEC}(m_t,a_t)}_{\text{remove self playback}}
\rightarrow
\underbrace{e_t}_{\substack{\text{speech, speaker,}\text{prosody, partial text}}}
\rightarrow
\underbrace{c_t=\pi(e_{\le t},s_t)}_{\text{interaction decision}}
\rightarrow
\underbrace{\text{Blue/custom agent}}_{\text{reason and use tools}}.
$$

The first implementation should therefore combine:

1. exact render-reference AEC;
2. generic speech probability plus target-speaker probability;
3. immediate reversible ducking on acoustic onset;
4. semantic confirmation before destructive cancellation;
5. a timestamped interaction/action lane;
6. a playback ledger recording what was actually heard;
7. a separate committed-tool ledger; and
8. optional endpoint anticipation only after the committed path is correct.
