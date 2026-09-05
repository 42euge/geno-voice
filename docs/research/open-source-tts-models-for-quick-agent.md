# Open-source TTS models for Geno Quick Agent

_Research snapshot: 2026-09-03. Primary sources only: official repositories,
model cards, source code, and measurements already captured in this repository._

## Executive recommendation

**Breeze-TTS-2 is not an eligible Quick Agent backend despite its strong voice
quality and streaming design.** Its source code is Apache-2.0, but its official
license explicitly says the model is not open source and limits the weights,
derivatives, and self-hosted outputs to research/non-commercial use. The 7.68
GB checkpoint's supported runtime requires Linux, CUDA, and at least a 12 GB
NVIDIA GPU. The released streaming class raises on any non-CUDA device, so it
does not run on Quick Agent's Apple-Silicon target without a new port. It is
worth using as a listening-quality and expressiveness reference, not as a
dependency.

Use **Kokoro 82M** as the first replacement candidate for
`AVSpeechSynthesizer`. It is the best fit for Quick Agent, not merely the best
small model on a leaderboard:

- Apache-2.0 code and weights;
- already runs locally in this repository on Apple Silicon;
- multiple English voices and an explicit speed control;
- warm synthesis was observed locally at about 162 ms from first LLM token to
  audio; and
- most importantly, its public pipeline exposes token `start_ts` and `end_ts`,
  preserving Quick Agent's synchronized word highlighting without a second
  forced-alignment model.

The implementation decision is really a choice between two Kokoro runtimes:

1. **Preserve macOS 13:** run the existing Python Kokoro pipeline behind a
   persistent local helper and preload it. The model itself is proven, but the
   measured 24.98 s cold load must be hidden at app launch or removed through
   packaging/runtime work.
2. **Raise the floor to macOS 14:** use
   [FluidAudio's native Swift/CoreML Kokoro path](https://github.com/FluidInference/FluidAudio/blob/main/Documentation/TTS/KokoroAne.md).
   It reports roughly 300 ms to synthesize five seconds of audio on an M1 and
   now exposes the model's per-token predicted durations. This is the cleanest
   native integration, but FluidAudio currently declares Swift tools 6.0 and
   macOS 14 while Quick Agent declares Swift tools 5.10 and macOS 13.

Run **PocketTTS** and **KittenTTS 0.8 Mini** beside Kokoro in a listening and
latency bake-off. PocketTTS is the best real-streaming challenger; KittenTTS is
the best small-download challenger. Neither public API currently returns word
timings, so neither should replace Kokoro unless its listening advantage is
large enough to justify approximate highlighting or a forced aligner.

Treat Breeze-TTS-2, Qwen3-TTS 0.6B, and Chatterbox Nano as quality-ceiling
experiments, not initial production choices. Their published assets are about
7.68 GB, 2.5 GB, and 3.0 GB respectively, and none solves Quick Agent's
word-timing requirement.

## Quick Agent's actual selection criteria

The current implementation and design establish stricter requirements than
"sounds good":

- entirely local/private after installation;
- Apple Silicon and the current macOS 13 deployment target;
- clearly more natural than the installed macOS voice;
- low time to first audible audio for short agent replies;
- immediate playback cancellation;
- several usable English voices and speaking-rate control;
- word timing that can be mapped to UTF-16 `NSRange` values for the existing
  highlight UI; and
- a distributable runtime and model footprint suitable for a signed Mac app.

The existing `SpeechPlaybackControlling` seam is already narrow enough for a
new backend: start, toggle, stop, speaking-state callbacks, and spoken-range
callbacks. The model choice does not require changing the coordinator or
answer rendering.

## Comparison

| Candidate | License and published size | Apple Silicon and latency evidence | Voices / rate | Public word timing | Quick Agent fit |
|---|---|---|---|---|---|
| **Breeze-TTS-2** | Apache-2.0 code, but weights, derivatives, and self-hosted outputs use a research/non-commercial license that expressly says it is not open source. Official checkpoint: 7.682 GB. | No supported Apple-Silicon path. Official requirements are Linux, CUDA, 7.7 GiB eager VRAM / 12 GB minimum GPU; the released streaming runtime rejects non-CUDA devices. Vendor reports <40 ms warmed TTFA and 0.32 RTF on H100, not on a Mac. | Voice cloning, reference-free voice design, natural-language direction of emotion/pace, and vocal-event tags. No numeric speed control. | **No.** Stream chunks contain PCM, codec-frame counts, and compute timing—not words, source spans, or alignment. The HTTP API strips even the internal timing metadata. | **Ineligible for the app.** Useful as an expressive quality ceiling or non-commercial H100 evaluation only. |
| **Kokoro 82M** | Apache-2.0. The main weight is 327 MB; the complete model repository is about 361 MB including voices and samples. | Existing Python path works locally. This repo observed a 24.98 s cold load and a 162 ms first-token-to-audio gap on one real run. FluidAudio's native path reports a ~0.3 s warm load, 3–11x real-time synthesis, and ~300 ms for 5 s of audio on M1. | Many English voices; explicit `speed`. FluidAudio's documented ANE fast path is more constrained than the Python voice set. | **Yes.** Python `KPipeline` attaches `start_ts`/`end_ts` to source tokens. FluidAudio exposes input token IDs and exact predicted acoustic-frame durations, though Quick Agent would still need to map phoneme tokens back to source-text ranges. | **Best first candidate.** The only option with a timing path already proven in this repo. |
| **PocketTTS** | MIT code; CC-BY-4.0 weights. 100M parameters. The current English model is 219 MB plus a roughly 4–8 MB voice state. | Official Python implementation is CPU-first: approximately 200 ms to its first streamed chunk and 6x real time on two M4 Air CPU cores. FluidAudio has a native CoreML port, but its current converted pack is 549 MB at int8 and the SDK requires macOS 14. | Many preset/reference voices and voice cloning. No documented speech-rate or pronunciation-control API. | **No documented output.** The stream yields audio, not source spans or token times. | **Best streaming challenger.** Requires approximate highlighting or post/parallel alignment. |
| **KittenTTS 0.8** | Apache-2.0. 15M/40M/80M ONNX variants; published footprints range from 25 MB for Nano int8 to about 80 MB for Mini. | Officially supports CPU inference on macOS. `generate_stream` yields one completed text chunk at a time; this is chunked synthesis, not evidence of codec-level first-audio streaming. No official TTFA figure is published. | Eight built-in voices; explicit `speed`. | **No.** The public ONNX wrapper returns only waveform output. | **Best footprint challenger.** Attractive if Mini wins listening tests and coarse highlighting is acceptable. |
| **Qwen3-TTS 0.6B CustomVoice** | Apache-2.0; official 0.6B CustomVoice assets total 2.493 GB. | Vendor reports end-to-end latency as low as 97 ms and calls the architecture streaming, but the official examples use CUDA. The public wrapper returns complete waveforms; its own docstring says `non_streaming_mode=false` only simulates streaming text input, not true streaming generation. No official MLX/CoreML or tested Apple-Silicon path is documented. | Nine voices; ten languages. The 0.6B CustomVoice model lacks the 1.7B model's instruction control. | **No.** Official generation methods return `(wavs, sample_rate)`. | **Research only.** Large, no supported native Mac route, no timing contract, and the published Python API cannot realize the headline streaming behavior. |
| **Chatterbox Nano** | MIT; 110M acoustic/text model, but its full official asset set is 2.997 GB because it also carries the shared S3Gen components. | Official code accepts CPU or MPS. Resemble claims 3x real time on eight CPU cores; no first-audio result is published. | Voice cloning from a reference clip and paralinguistic tags such as `[laugh]`; no direct rate control. | **No.** `generate` returns a completed waveform. | **Research only.** Expressive, but its actual distribution size and reference-voice workflow are poor fits for a small utility. |

Published sizes above count the model assets required by the referenced
official repositories, not Python, PyTorch, ONNX Runtime, CoreML compilation
caches, or app packaging. Those runtime costs must be measured separately.

## Candidate details

### Breeze-TTS-2: an impressive but ineligible quality ceiling

[Breeze-TTS-2](https://huggingface.co/BreezeBlue/Breeze-TTS-2) is the most
expressive candidate examined here. It can clone a reference speaker, design a
voice from a natural-language description, direct a cloned voice's emotion and
pace, and synthesize inline events such as laughs, coughs, and sighs. Its
official model card says it ranks first among open-weight systems on the
Artificial Analysis TTS leaderboard. That makes it worth listening to before
setting a quality target for Quick Agent.

Its product fit is nevertheless unambiguous:

- The model card and
  [weight license](https://huggingface.co/BreezeBlue/Breeze-TTS-2/blob/main/LICENSE)
  distinguish Apache-2.0 inference code from the model materials. The license
  states that it is **not an open-source license**, prohibits commercial use
  of the weights and self-hosted outputs, and requires a separate written
  license for production use.
- The official checkpoint contains 6.967 GB of BF16 model shards, a 682 MB
  audio tokenizer, and a 33 MB text tokenizer: 7,682,237,079 bytes of LFS
  assets in total.
- The official requirements are Linux, Python 3.10+, CUDA, about 7.7 GiB of
  eager VRAM, and a 12 GB minimum NVIDIA GPU. Its under-40-ms TTFA and 0.32 RTF
  are warmed fast-path H100 measurements.
- Although the general loader falls back to CPU when CUDA is absent, both the
  official CLI and HTTP API construct
  [`FastBreezeStreamingRuntime`](https://github.com/breezeblue-ai/breeze-tts/blob/main/models/fast_streaming.py),
  whose constructor raises unless the device is CUDA. There is no documented
  MPS, MLX, CoreML, or macOS path.

The streaming implementation itself is real. It generates 12.5 Hz codec
frames autoregressively and decodes/yields one frame per fast-path chunk or two
frames in the eager codec path. The
[`FastStreamingChunk`](https://github.com/breezeblue-ai/breeze-tts/blob/main/models/fast_streaming.py)
contains waveform samples, sample rate, codec-frame count, a final flag, and
stage timing. It contains no generated word, source range, or text/audio
alignment. The
[`/v1/audio/speech` API](https://github.com/breezeblue-ai/breeze-tts/blob/main/breeze_infer/api.py)
streams only raw 24 kHz PCM bytes and sample-format headers, discarding the
chunk metadata. It serializes inference through a single process lock and has
no explicit cancel endpoint; a disconnected response can close the generator
and release model state, but that is a transport behavior rather than a
published cancellation contract.

Natural-language direction can request "speak slowly," but it is not the
deterministic numeric speed control Quick Agent currently exposes. The output
also lacks the word spans required by `onSpokenRangeChanged`.

Breeze therefore has two legitimate roles here:

1. listen to its public examples as the quality/expressiveness ceiling; or
2. for research evaluation only, render the shared bake-off corpus on a
   suitable NVIDIA host and keep those restricted outputs out of any product
   distribution.

Its hosted commercial API avoids the self-hosted output restriction, but it
would send answer text off-device and directly violates this project's
offline/private requirement.

### 1. Kokoro 82M

The official [Kokoro inference library](https://github.com/hexgrad/kokoro)
describes the model as 82M parameters with Apache-licensed weights. The
[model repository](https://huggingface.co/hexgrad/Kokoro-82M) contains a
327,212,226-byte main checkpoint and small per-voice tensors.

Kokoro has a decisive integration advantage. Its
[pipeline source](https://github.com/hexgrad/kokoro/blob/main/kokoro/pipeline.py)
joins the model's predicted phoneme durations to `MToken` objects and assigns
`start_ts` and `end_ts`. Geno-voice already consumes those fields in
[`examples/_chat_tts.py`](../../examples/_chat_tts.py), offsets timings across
chunks, skips untimed punctuation safely, and drives cancellation-aware
playback from them. This is working product evidence, not a proposed feature.

The local performance evidence is mixed but actionable. The documented real
run in [the human voice turn explainer](../human-voice-turn-explainer.md)
measured:

- 24,984 ms to load Kokoro cold;
- 162 ms for observed synthesis / first-token-to-audio; and
- 322 ms from barge-in detection to halted playback for the whole current
  pipeline.

The cold load makes a launch-on-first-speech Python process unacceptable. A
persistent helper can load during app startup and keep the model warm, but it
adds Python/PyTorch packaging and process-lifecycle work.

FluidAudio is the promising native route. Its current
[`KokoroAneSynthesisResult`](https://github.com/FluidInference/FluidAudio/blob/main/Sources/FluidAudio/TTS/KokoroAne/Pipeline/KokoroAneSynthesizer%2BTypes.swift)
returns raw samples, input token IDs, and `predictedDurations`; the source says
these are the exact durations used to build the downstream alignment. That
removes the need to align synthesized audio again. It does not by itself
provide the answer's character ranges, so an adapter still has to preserve a
text/word-to-phoneme mapping and aggregate token durations into `NSRange`s.

FluidAudio's current
[`Package.swift`](https://github.com/FluidInference/FluidAudio/blob/main/Package.swift)
sets `.macOS(.v14)` and Swift tools 6.0. Its Kokoro ANE documentation also
notes an approximately 20 s first-ever ANE compilation on M1, a ~0.3 s warm
load, a 510-phoneme per-call limit, and no built-in chunker. Those are
engineering constraints, not model disqualifiers, but they prevent a drop-in
dependency while Quick Agent supports macOS 13.

### 2. PocketTTS

Kyutai's [official repository](https://github.com/kyutai-labs/pocket-tts)
publishes the strongest interaction-oriented claims in this group: 100M
parameters, CPU operation, two CPU cores, audio streaming, about 200 ms to the
first chunk, and about 6x real time on a MacBook Air M4. Its
[`generate_audio_stream`](https://github.com/kyutai-labs/pocket-tts/blob/main/pocket_tts/models/tts_model.py)
actually yields decoded waveform chunks as they become available, so
progressive playback and cancellation do not depend on sentence boundaries.

The tradeoffs are equally concrete:

- code is MIT, while the
  [published weights](https://huggingface.co/kyutai/pocket-tts) are CC-BY-4.0;
- the public output is waveform chunks, with no word/token alignment data;
- the raw-text-token architecture has no IPA layer for a custom pronunciation
  dictionary; and
- the official public API does not document a speaking-rate control.

FluidAudio also implements PocketTTS as a persistent streaming session with
cancel support. Its own
[PocketTTS documentation](https://github.com/FluidInference/FluidAudio/blob/main/Documentation/TTS/PocketTTS.md)
confirms that pronunciation overrides and phoneme tags are unavailable. The
native conversion is substantially larger than Kyutai's Python weights and is
still blocked by FluidAudio's macOS 14 floor.

PocketTTS should advance if listeners strongly prefer it to Kokoro. The price
would be one of: sentence-level highlighting, proportional timing estimates,
or an offline forced aligner running beside playback. None is as simple or as
exact as consuming Kokoro's generation-time durations.

### 3. KittenTTS 0.8

The [official KittenTTS repository](https://github.com/KittenML/KittenTTS)
ships Apache-2.0 ONNX code and weights, eight voices, speed control, 24 kHz
output, and CPU support on macOS. Its 80M Mini model is about 81.5 MB including
voices; the 15M Nano int8 option is advertised at 25 MB.

The source-level caveat matters. The public
[`generate_stream`](https://github.com/KittenML/KittenTTS/blob/main/kittentts/onnx_model.py)
splits text and runs complete ONNX synthesis once per text chunk before
yielding that chunk. It is useful for long answers and cancellation between
chunks, but it is not evidence of a 200 ms streaming TTFA. The ONNX wrapper
only uses the first model output as audio and publishes no token-duration or
alignment object.

Mini, not Nano, should enter the first listening test: the point is to find a
voice clearly better than AVSpeech, and the 80 MB model is already small enough
to make footprint a secondary concern. Nano becomes relevant only if Mini
first clears the quality bar.

### 4. Qwen3-TTS and Chatterbox Nano

Both deserve listening samples because they explore a more expressive quality
ceiling, but neither should dictate the first app architecture.

[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) is Apache-2.0, multilingual,
and advertises 97 ms best-case latency. The official 0.6B model repository,
however, contains 1.811 GB of model weights plus a 682 MB speech tokenizer.
Official examples use CUDA and FlashAttention. More importantly, the current
[`Qwen3TTSModel` wrapper](https://github.com/QwenLM/Qwen3-TTS/blob/main/qwen_tts/inference/qwen3_tts_model.py)
returns full waveform lists and explicitly says its apparent streaming mode is
only a simulation of streaming text input. The published latency is therefore
not a usable Quick Agent API contract today.

[Chatterbox Nano](https://github.com/resemble-ai/chatterbox) is MIT and its
official source accepts `cpu` and `mps`. Nano itself is 110M parameters, but
the [official checkpoint repository](https://huggingface.co/ResembleAI/chatterbox-nano)
is almost 3 GB once the shared decoder and voice encoder are included. The
official example asks for a reference voice clip, `generate` returns a complete
waveform, and no word timing or direct rate control is exposed. Expressive tags
are attractive, but not enough to offset those product costs.

## Candidates not recommended for this bake-off

- **F5-TTS:** code is MIT, but the official repository states that pretrained
  weights are CC-BY-NC because of their training data. That is not a safe
  default for a distributable product.
- **Fish Speech:** the official Fish Audio Research License requires a
  separate agreement for any commercial use. It is not open-source in the
  practical sense needed here.
- **Supertonic 3:** compact 99M ONNX model with Swift examples, but the weights
  use OpenRAIL-M rather than an OSI software license, and the official
  repository says open-source model development/support is ending. It can be
  revisited if "open weights with use restrictions" is acceptable.
- **Dia 1.6B:** Apache-2.0, but its official repository says short inputs under
  five seconds sound unnatural and that it has only been tested on CUDA GPUs.
  That is almost the inverse of Quick Agent's workload.
- **Piper:** still an efficient baseline, but the original MIT repository is
  archived and points to the actively maintained GPL-3 successor. Voice
  licenses also vary. It is more useful as a footprint/latency floor than as a
  likely naturalness winner.

## Proposed bake-off

Do not choose from vendor samples. Build one adapter harness and render the
same corpus through:

1. the current selected `AVSpeechSynthesisVoice` at the current preference;
2. Kokoro `af_heart` plus one male English voice;
3. PocketTTS with two redistributable English voice states; and
4. KittenTTS Mini with its two best voices after a quick screening pass.

Use 30–50 prompts covering one-word answers, ordinary conversational replies,
long sentences, punctuation, numerals, dates, acronyms, product/model names,
URLs, quoted text, and intentionally awkward pronunciation. Record:

- blind pairwise naturalness preference against AVSpeech;
- intelligibility/pronunciation failures;
- cold load, warm load, and first-audio p50/p95;
- synthesis real-time factor and peak resident memory;
- installed model/runtime size;
- cancellation-to-silence latency;
- voice-to-voice consistency and useful rate range; and
- word-highlight onset error against manually checked or forced-aligned
  reference timestamps.

Recommended gates for advancing a backend are:

- listeners clearly prefer it to AVSpeech on conversational prompts;
- warm first audio stays below 300 ms p95 for short replies;
- cancellation becomes inaudible within 100 ms after the stop request;
- no network access is required during normal operation; and
- highlighting is exact enough to avoid visibly leading or lagging speech.

The bake-off should report Python and native runtimes separately. A model can
win on voice quality while its current runtime loses on launch time, footprint,
or OS compatibility.

## Decision

The next engineering spike should be **Kokoro playback behind the existing
`SpeechPlaybackControlling` interface**, retaining AVSpeech as fallback. Before
choosing the runtime, make one product decision explicit:

- if macOS 13 support is non-negotiable, prototype a warm persistent Python
  helper and measure its signed-app footprint and launch behavior;
- if macOS 14 is acceptable, prototype FluidAudio Kokoro and the mapping from
  predicted phoneme durations to the answer's UTF-16 word ranges.

In parallel, render PocketTTS and KittenTTS Mini samples. Only a decisive blind
listening win should justify giving up Kokoro's native timing advantage.
