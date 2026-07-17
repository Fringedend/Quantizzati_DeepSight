#!/usr/bin/env python3
"""Smoke test locale di JPEG, MP4, OpenCV e FFmpeg senza caricare modelli AI."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

RADICE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RADICE / "src"))

from PIL import Image  # noqa: E402
import imageio_ffmpeg  # noqa: E402
import config  # noqa: E402
import processor  # noqa: E402


def main() -> int:
    with tempfile.TemporaryDirectory() as cartella:
        base = Path(cartella)
        foto = base / "foto.jpg"
        video = base / "video.mp4"
        config.DIR_ANTEPRIME = str(base / "thumbnails")
        Path(config.DIR_ANTEPRIME).mkdir()
        Image.new("RGB", (160, 120), "navy").save(foto, "JPEG")
        assert processor.crea_anteprima(str(foto), "image")

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        comando = [
            ffmpeg, "-y",
            "-f", "lavfi", "-i", "testsrc=size=160x120:rate=5",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000",
            "-t", "1", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", str(video),
        ]
        subprocess.run(comando, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
        proprieta = processor.ottieni_proprieta_video(str(video))
        assert proprieta and proprieta["larghezza"] == 160 and proprieta["altezza"] == 120
        audio = processor.estrai_audio_video(str(video))
        assert audio and Path(audio).is_file()
        Path(audio).unlink()
    print("Smoke test media: JPEG, MP4, OpenCV e FFmpeg OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
