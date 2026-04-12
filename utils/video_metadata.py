from __future__ import annotations

from pathlib import Path
from typing import Any

from .models import VideoMetadata


def probe_video_metadata(video_path: str) -> VideoMetadata:
    resolved_path = Path(video_path).resolve(strict=False)
    if not resolved_path.is_file():
        raise ValueError(f"Video path not found: {video_path}")

    try:
        import cv2
    except Exception as exc:
        raise RuntimeError("Video metadata probing requires OpenCV. Install `opencv-python-headless`.") from exc

    capture = cv2.VideoCapture(str(resolved_path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open video for metadata probing: {resolved_path}")

        fps = _coerce_positive_float(capture.get(cv2.CAP_PROP_FPS))
        total_frames = _coerce_positive_int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = None
        if fps and total_frames:
            duration_sec = float(total_frames) / float(fps)

        return VideoMetadata(
            source_video_path=str(resolved_path),
            fps=fps,
            total_frames=total_frames,
            duration_sec=duration_sec,
        )
    finally:
        capture.release()


def _coerce_positive_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number <= 0:
        return None
    return number


def _coerce_positive_int(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    if number <= 0:
        return None
    return number
