from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .models import SampledKeyframe, VideoMetadata

MAX_SAMPLED_FRAMES = 60


@dataclass(frozen=True)
class VideoFrameSample:
    keyframe: SampledKeyframe
    image: Any


def sample_video_frames(video_path: str, metadata: VideoMetadata) -> tuple[VideoFrameSample, ...]:
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError("Video frame sampling requires OpenCV. Install `opencv-python-headless`.") from exc

    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open video for frame sampling: {video_path}")

        frame_indices = _choose_frame_indices(metadata)
        samples: list[VideoFrameSample] = []
        for sample_index, frame_index in enumerate(frame_indices, start=1):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame_bgr = capture.read()
            if not ok or frame_bgr is None:
                continue
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            timestamp_sec = None
            if metadata.fps:
                timestamp_sec = round(float(frame_index) / float(metadata.fps), 6)
            samples.append(
                VideoFrameSample(
                    keyframe=SampledKeyframe(
                        index=sample_index,
                        frame_index=int(frame_index),
                        timestamp_sec=timestamp_sec,
                    ),
                    image=frame_rgb,
                )
            )
        return tuple(samples)
    finally:
        capture.release()


def _choose_frame_indices(metadata: VideoMetadata) -> list[int]:
    fps = metadata.fps
    total_frames = metadata.total_frames
    duration_sec = metadata.duration_sec

    if fps and total_frames and duration_sec:
        last_frame = max(0, int(total_frames) - 1)
        if duration_sec <= MAX_SAMPLED_FRAMES:
            count = max(1, min(MAX_SAMPLED_FRAMES, int(duration_sec + 0.999999)))
            return sorted({min(last_frame, int(round(second * fps))) for second in range(count)})
        return _evenly_spaced_indices(last_frame, MAX_SAMPLED_FRAMES)

    if total_frames:
        last_frame = max(0, int(total_frames) - 1)
        count = min(MAX_SAMPLED_FRAMES, int(total_frames))
        if total_frames <= MAX_SAMPLED_FRAMES:
            return list(range(count))
        return _evenly_spaced_indices(last_frame, count)

    return [0]


def _evenly_spaced_indices(last_frame: int, count: int) -> list[int]:
    if count <= 1 or last_frame <= 0:
        return [0]
    return sorted({int(round((last_frame * index) / (count - 1))) for index in range(count)})
