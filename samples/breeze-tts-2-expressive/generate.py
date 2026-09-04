#!/usr/bin/env python3
"""Generate and validate the Breeze TTS 2 expressive sample set."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import websockets

from geno_voice.endpoint.transports.wire import decode_audio_envelope


EXPECTED_RATE = 24_000
EXPECTED_CHANNELS = 1
EXPECTED_SAMPLE_WIDTH = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).with_name("manifest.json"),
    )
    parser.add_argument(
        "--endpoint",
        help="WebSocket URL; defaults to the endpoint recorded in the manifest",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate WAVs that already exist",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=240.0,
        help="Maximum seconds to wait for one synthesis event",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate existing WAVs and refresh their measured metadata",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest.get("samples"), list) or not manifest["samples"]:
        raise ValueError("manifest must contain a non-empty samples array")
    return manifest


def inspect_wav(path: Path) -> dict[str, Any]:
    with wave.open(str(path), "rb") as source:
        channels = source.getnchannels()
        sample_width = source.getsampwidth()
        sample_rate = source.getframerate()
        frame_count = source.getnframes()
        compression = source.getcomptype()

    if channels != EXPECTED_CHANNELS:
        raise ValueError(f"{path.name}: expected mono audio, got {channels} channels")
    if sample_width != EXPECTED_SAMPLE_WIDTH:
        raise ValueError(
            f"{path.name}: expected 16-bit audio, got {sample_width * 8}-bit"
        )
    if sample_rate != EXPECTED_RATE:
        raise ValueError(
            f"{path.name}: expected {EXPECTED_RATE} Hz, got {sample_rate} Hz"
        )
    if compression != "NONE":
        raise ValueError(f"{path.name}: expected uncompressed PCM, got {compression}")
    if frame_count == 0:
        raise ValueError(f"{path.name}: WAV contains no audio frames")

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "duration_seconds": round(frame_count / sample_rate, 3),
        "frame_count": frame_count,
        "file_size_bytes": path.stat().st_size,
        "sha256": digest,
    }


def write_wav(path: Path, pcm: bytes) -> None:
    if not pcm or len(pcm) % EXPECTED_SAMPLE_WIDTH:
        raise ValueError(f"{path.name}: invalid PCM payload length {len(pcm)}")
    temporary = path.with_suffix(path.suffix + ".part")
    with wave.open(str(temporary), "wb") as output:
        output.setnchannels(EXPECTED_CHANNELS)
        output.setsampwidth(EXPECTED_SAMPLE_WIDTH)
        output.setframerate(EXPECTED_RATE)
        output.writeframes(pcm)
    os.replace(temporary, path)


async def receive_event(socket: Any, timeout: float) -> str | bytes:
    return await asyncio.wait_for(socket.recv(), timeout=timeout)


async def synthesize_sample(
    socket: Any,
    sample: dict[str, Any],
    timeout: float,
    *,
    first_sequence: int,
) -> tuple[bytes, int]:
    request_id = sample["id"]
    await socket.send(
        json.dumps(
            {
                "type": "speak",
                "request_id": request_id,
                "text": sample["text"],
                "instruction": sample["instruction"],
            },
            ensure_ascii=False,
        )
    )

    chunks: list[bytes] = []
    expected_sequence = first_sequence
    while True:
        message = await receive_event(socket, timeout)
        if isinstance(message, bytes):
            header, pcm = decode_audio_envelope(message)
            if header.get("request_id") != request_id:
                raise RuntimeError(
                    f"{request_id}: received audio for {header.get('request_id')}"
                )
            if header.get("sample_rate") != EXPECTED_RATE:
                raise RuntimeError(
                    f"{request_id}: endpoint emitted {header.get('sample_rate')} Hz"
                )
            if header.get("encoding") != "pcm_s16le":
                raise RuntimeError(
                    f"{request_id}: endpoint emitted {header.get('encoding')}"
                )
            if header.get("sequence") != expected_sequence:
                raise RuntimeError(
                    f"{request_id}: expected chunk {expected_sequence}, "
                    f"received {header.get('sequence')}"
                )
            if len(pcm) % EXPECTED_SAMPLE_WIDTH:
                raise RuntimeError(f"{request_id}: endpoint emitted partial PCM frame")
            chunks.append(pcm)
            expected_sequence += 1
            continue

        event = json.loads(message)
        event_request_id = event.get("request_id")
        if event_request_id not in {None, request_id}:
            raise RuntimeError(
                f"{request_id}: received event for {event_request_id}: {event}"
            )
        if event.get("type") == "error":
            raise RuntimeError(
                f"{request_id}: {event.get('code')}: {event.get('message')}"
            )
        if event.get("type") == "completed":
            if not chunks:
                raise RuntimeError(f"{request_id}: completed without audio")
            return b"".join(chunks), expected_sequence


def update_manifest(path: Path, manifest: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(manifest, output, ensure_ascii=False, indent=2)
        output.write("\n")
    os.replace(temporary, path)


async def generate(args: argparse.Namespace, manifest: dict[str, Any]) -> None:
    manifest_path = args.manifest.resolve()
    output_dir = manifest_path.parent
    endpoint = args.endpoint or manifest["endpoint"]
    manifest["endpoint"] = endpoint

    if args.validate_only:
        for sample in manifest["samples"]:
            path = output_dir / sample["filename"]
            sample["output"] = inspect_wav(path)
            print(
                f"validated {path.name}: "
                f"{sample['output']['duration_seconds']:.3f}s"
            )
        update_manifest(manifest_path, manifest)
        return

    async with websockets.connect(
        endpoint,
        proxy=None,
        max_size=None,
        open_timeout=20,
        ping_timeout=30,
    ) as socket:
        ready_message = await receive_event(socket, args.timeout)
        if isinstance(ready_message, bytes):
            raise RuntimeError("endpoint sent binary audio before its ready event")
        ready = json.loads(ready_message)
        if ready.get("type") != "ready":
            raise RuntimeError(f"expected ready event, received {ready}")
        print(f"connected: session={ready.get('session_id')} endpoint={endpoint}")
        next_sequence = 0

        for index, sample in enumerate(manifest["samples"], start=1):
            output_path = output_dir / sample["filename"]
            if output_path.exists() and not args.overwrite:
                print(f"[{index:02d}/{len(manifest['samples'])}] keeping {output_path.name}")
            else:
                print(
                    f"[{index:02d}/{len(manifest['samples'])}] "
                    f"generating {output_path.name}",
                    flush=True,
                )
                pcm, next_sequence = await synthesize_sample(
                    socket,
                    sample,
                    args.timeout,
                    first_sequence=next_sequence,
                )
                write_wav(output_path, pcm)
            sample["output"] = inspect_wav(output_path)
            print(
                f"             {sample['output']['duration_seconds']:.3f}s, "
                f"{sample['output']['file_size_bytes']} bytes",
                flush=True,
            )
            update_manifest(manifest_path, manifest)

    manifest["generated_at"] = datetime.now(timezone.utc).isoformat()
    update_manifest(manifest_path, manifest)


def main() -> int:
    args = parse_args()
    try:
        manifest = load_manifest(args.manifest.resolve())
        asyncio.run(generate(args, manifest))
    except (OSError, ValueError, RuntimeError, asyncio.TimeoutError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
