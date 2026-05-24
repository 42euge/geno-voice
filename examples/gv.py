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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"


def cmd_bench(args):
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


def main():
    parser = argparse.ArgumentParser(prog="gv", description="geno-voice CLI")
    sub = parser.add_subparsers(dest="command")

    bench = sub.add_parser("bench", help="Batch mode — transcribe after silence")
    bench.add_argument("--model", default=DEFAULT_MODEL)

    stream = sub.add_parser("stream", help="Streaming mode — live progressive transcription")
    stream.add_argument("--model", default=DEFAULT_MODEL)

    talk = sub.add_parser("talk", help="Talk mode — STT → NLP → canned response → TTS")
    talk.add_argument("--model", default=DEFAULT_MODEL)
    talk.add_argument("--voice", default="af_heart", help="TTS voice (default: af_heart)")
    talk.add_argument("--speed", type=float, default=1.0, help="TTS speed (default: 1.0)")

    chat = sub.add_parser("chat", help="Chat mode — STT → LLM (litellm) → TTS")
    chat.add_argument("--model", default=DEFAULT_MODEL)
    chat.add_argument("--voice", default="af_heart", help="TTS voice (default: af_heart)")
    chat.add_argument("--speed", type=float, default=1.0, help="TTS speed (default: 1.0)")

    args = parser.parse_args()

    if args.command == "bench":
        cmd_bench(args)
    elif args.command == "stream":
        cmd_stream(args)
    elif args.command == "talk":
        cmd_talk(args)
    elif args.command == "chat":
        cmd_chat(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
