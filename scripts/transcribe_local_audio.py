#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

try:
    import torch
    import whisper
except ModuleNotFoundError as exc:
    missing_name = exc.name or "dependency"
    raise SystemExit(
        f"Missing Python dependency: {missing_name}. "
        "Run `python3 -m pip install -e .` in this repository first."
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIO = REPO_ROOT / "youtube_audio_first_2min.mp3"


def pick_device(requested_device: Optional[str]) -> str:
    if requested_device:
        return requested_device
    if torch.cuda.is_available():
        return "cuda"

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"

    return "cpu"


def json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe a local audio file with Whisper and save txt/json outputs."
    )
    parser.add_argument(
        "--audio",
        type=Path,
        default=DEFAULT_AUDIO,
        help=f"Audio file to transcribe. Default: {DEFAULT_AUDIO}",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="base",
        choices=whisper.available_models(),
        help="Whisper model name.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Language code such as de, en, zh. Leave empty to auto-detect.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="transcribe",
        choices=("transcribe", "translate"),
        help="Use `translate` only with multilingual models, not `turbo`.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Force `cuda`, `mps`, or `cpu`. Default: auto-select.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for output files. Default: next to the audio file.",
    )
    parser.add_argument(
        "--download-root",
        type=Path,
        default=None,
        help="Optional Whisper model cache directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audio_path = args.audio.expanduser().resolve()
    if not audio_path.exists():
        raise SystemExit(f"Audio file not found: {audio_path}")
    if args.task == "translate" and args.model == "turbo":
        raise SystemExit("`turbo` is not reliable for translation. Use `base` or larger.")

    output_dir = (args.output_dir or audio_path.parent).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device(args.device)
    model = whisper.load_model(
        args.model,
        device=device,
        download_root=str(args.download_root.expanduser().resolve())
        if args.download_root
        else None,
    )

    # Whisper 仅在 CUDA 上建议开启 fp16；CPU/MPS 统一走 fp32，兼顾兼容性和稳定性。
    result = model.transcribe(
        str(audio_path),
        task=args.task,
        language=args.language,
        word_timestamps=True,
        temperature=0.0,
        verbose=False,
        fp16=device == "cuda",
    )

    stem = audio_path.stem
    text_path = output_dir / f"{stem}.whisper.txt"
    json_path = output_dir / f"{stem}.whisper.json"

    text_path.write_text(result["text"].strip() + "\n", encoding="utf-8")
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )

    print(f"audio={audio_path}")
    print(f"model={args.model}")
    print(f"device={device}")
    print(f"language={result['language']}")
    print(f"text_file={text_path}")
    print(f"json_file={json_path}")
    print()
    print(result["text"].strip())


if __name__ == "__main__":
    main()
