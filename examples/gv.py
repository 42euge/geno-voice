#!/usr/bin/env python3
"""geno-voice CLI.

Usage:
    gv bench              # batch mode — wait for silence, transcribe, show timing
    gv stream             # streaming mode — live progressive transcription
    gv talk               # talk mode — STT → NLP → canned response → TTS
    gv chat               # chat mode — STT → LLM (litellm) → TTS
    gv <cmd> --model ...  # override STT model
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"

# TTS speech-rate multiplier bounds (the kokoro `speed` parameter). The engine
# treats this as a wall-clock multiplier on synthesis, so values <= 0 are
# nonsensical (zero / negative-rate speech) and very large values produce
# unintelligible output. The accepted window matches kokoro's documented
# practical range; the parser rejects anything outside it with the usual
# argparse SystemExit(2) instead of forwarding garbage to the TTS engine.
SPEED_MIN = 0.5
SPEED_MAX = 2.0


def speed_type(raw):
    """Argparse ``type`` for ``--speed``: parse to float and bound-check.

    Pure and side-effect-free so it can be unit-tested directly. Raises
    :class:`argparse.ArgumentTypeError` (which argparse renders as
    ``SystemExit(2)``) when ``raw`` is not a number or falls outside
    ``[SPEED_MIN, SPEED_MAX]``. NaN is rejected explicitly — it compares
    false against both bounds, so without the guard the range message would
    misleadingly name the bounds rather than the real problem.
    """
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"speed must be a number, got {raw!r}")
    if value != value:  # NaN is unordered; name it directly.
        raise argparse.ArgumentTypeError("speed must be a number, got nan")
    if not (SPEED_MIN <= value <= SPEED_MAX):
        raise argparse.ArgumentTypeError(
            f"speed must be between {SPEED_MIN} and {SPEED_MAX}, got {value}"
        )
    return value


# Kokoro voice-id grammar: a single-letter language code, a one-letter gender
# (`f`/`m`), an underscore, then a lowercase a-z name. The curated American set
# lives in `tts/kokoro_engine.py:VOICES` (af_*/am_*), but kokoro also ships
# other-language packs (e.g. `bf_emma`, British female), so a strict membership
# check would wrongly reject legitimate voices. We validate the *format*
# instead — the same "close the garbage-in path at the parser" goal as
# `speed_type` (iter-182) — rejecting empty/whitespace/malformed ids before
# they reach the TTS engine, while staying permissive about the name itself.
VOICE_RE = re.compile(r"^[a-z][fm]_[a-z]+$")


def voice_type(raw):
    """Argparse ``type`` for ``--voice``: validate the kokoro id format.

    Pure and side-effect-free so it can be unit-tested directly. Raises
    :class:`argparse.ArgumentTypeError` (rendered by argparse as
    ``SystemExit(2)``) when ``raw`` does not match the kokoro voice-id
    grammar ``<lang><gender>_<name>`` (e.g. ``af_heart``, ``bf_emma``). This
    catches empty strings, leading/trailing whitespace, and obvious typos
    before they reach the engine, without pinning to the American-only
    curated list.
    """
    if not isinstance(raw, str) or not VOICE_RE.match(raw):
        raise argparse.ArgumentTypeError(
            "voice must look like '<lang><gender>_<name>' "
            f"(e.g. af_heart, bf_emma), got {raw!r}"
        )
    return raw


def cmd_bench(args):
    # bench is a legacy argv-driven entrypoint: it parses its own sys.argv
    # rather than taking kwargs, so we rebuild argv here. Only forward
    # --model when it differs from the default so the bench parser keeps
    # using its own default otherwise.
    from mic_bench import main as bench_main
    sys.argv = ["gv bench"]
    if args.model != DEFAULT_MODEL:
        sys.argv.extend(["--model", args.model])
    bench_main()


def cmd_stream(args):
    from mic_stream import run_stream
    run_stream(model_repo=args.model)


def cmd_talk(args):
    from mic_talk import run_talk
    run_talk(model_repo=args.model, voice=args.voice, speed=args.speed)


def cmd_chat(args):
    from mic_chat import run_chat
    run_chat(model_repo=args.model, voice=args.voice, speed=args.speed)


# Command-name → handler. Injectable so dispatch() can be unit-tested with
# stub handlers instead of importing the audio modules.
DEFAULT_HANDLERS = {
    "bench": cmd_bench,
    "stream": cmd_stream,
    "talk": cmd_talk,
    "chat": cmd_chat,
}


def build_parser():
    """Construct the gv argument parser.

    Pure: no I/O, no audio imports. The returned parser is safe to
    exercise from tests with ``parse_args([...])``.
    """
    parser = argparse.ArgumentParser(prog="gv", description="geno-voice CLI")
    sub = parser.add_subparsers(dest="command")

    bench = sub.add_parser("bench", help="Batch mode — transcribe after silence")
    bench.add_argument("--model", default=DEFAULT_MODEL)

    stream = sub.add_parser("stream", help="Streaming mode — live progressive transcription")
    stream.add_argument("--model", default=DEFAULT_MODEL)

    talk = sub.add_parser("talk", help="Talk mode — STT → NLP → canned response → TTS")
    talk.add_argument("--model", default=DEFAULT_MODEL)
    talk.add_argument(
        "--voice",
        type=voice_type,
        default="af_heart",
        help="TTS voice id, e.g. af_heart / bf_emma (default: af_heart)",
    )
    talk.add_argument(
        "--speed",
        type=speed_type,
        default=1.0,
        help=f"TTS speed in [{SPEED_MIN}, {SPEED_MAX}] (default: 1.0)",
    )

    chat = sub.add_parser("chat", help="Chat mode — STT → LLM (litellm) → TTS")
    chat.add_argument("--model", default=DEFAULT_MODEL)
    chat.add_argument(
        "--voice",
        type=voice_type,
        default="af_heart",
        help="TTS voice id, e.g. af_heart / bf_emma (default: af_heart)",
    )
    chat.add_argument(
        "--speed",
        type=speed_type,
        default=1.0,
        help=f"TTS speed in [{SPEED_MIN}, {SPEED_MAX}] (default: 1.0)",
    )

    return parser


def dispatch(args, parser, *, handlers=None):
    """Route parsed args to the matching command handler.

    Returns the process exit code: 0 on a dispatched command, 1 when no
    (or an unknown) command was given. Handlers are injectable for
    testing; the default map wires the real audio entrypoints.
    """
    handlers = DEFAULT_HANDLERS if handlers is None else handlers

    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    handler(args)
    return 0


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return dispatch(args, parser)


if __name__ == "__main__":
    sys.exit(main())
