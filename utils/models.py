from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class VideoMetadata:
    source_video_path: str
    fps: float | None
    total_frames: int | None = None
    duration_sec: float | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "source_video_path": self.source_video_path,
            "fps": self.fps,
            "total_frames": self.total_frames,
            "duration_sec": self.duration_sec,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any] | None) -> "VideoMetadata | None":
        if not payload:
            return None
        return cls(
            source_video_path=str(payload.get("source_video_path") or ""),
            fps=_coerce_optional_float(payload.get("fps")),
            total_frames=_coerce_optional_int(payload.get("total_frames")),
            duration_sec=_coerce_optional_float(payload.get("duration_sec")),
        )


@dataclass(frozen=True)
class SampledKeyframe:
    index: int
    frame_index: int
    timestamp_sec: float | None

    def to_payload(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "frame_index": self.frame_index,
            "timestamp_sec": self.timestamp_sec,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "SampledKeyframe":
        return cls(
            index=int(payload.get("index") or 0),
            frame_index=int(payload.get("frame_index") or 0),
            timestamp_sec=_coerce_optional_float(payload.get("timestamp_sec")),
        )


@dataclass(frozen=True)
class VideoDescriptionRecord:
    source_video_path: str
    filename: str
    status: str
    metadata: VideoMetadata | None = None
    keyframes: tuple[SampledKeyframe, ...] = ()
    description: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    error: str | None = None
    cache_key: str | None = None

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "source_video_path": self.source_video_path,
            "filename": self.filename,
            "status": self.status,
            "metadata": self.metadata.to_payload() if self.metadata else None,
            "sampled_keyframes": [keyframe.to_payload() for keyframe in self.keyframes],
            "description": self.description,
            "warnings": list(self.warnings),
            "error": self.error,
            "cache_key": self.cache_key,
        }
        return payload

    def to_error_payload(self) -> dict[str, Any]:
        return {
            "source_video_path": self.source_video_path,
            "filename": self.filename,
            "status": self.status,
            "error": self.error,
            "warnings": list(self.warnings),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "VideoDescriptionRecord":
        keyframes_payload = payload.get("sampled_keyframes") or payload.get("keyframes") or []
        return cls(
            source_video_path=str(payload.get("source_video_path") or ""),
            filename=str(payload.get("filename") or ""),
            status=str(payload.get("status") or "failed"),
            metadata=VideoMetadata.from_payload(payload.get("metadata")),
            keyframes=tuple(SampledKeyframe.from_payload(item) for item in keyframes_payload),
            description=dict(payload.get("description") or {}),
            warnings=tuple(str(item) for item in payload.get("warnings") or ()),
            error=str(payload.get("error")) if payload.get("error") is not None else None,
            cache_key=str(payload.get("cache_key")) if payload.get("cache_key") is not None else None,
        )


@dataclass(frozen=True)
class JoyCaptionShotDescriptionResult:
    source_path: str
    generated_at: str
    model_id: str
    caption_max_tokens: int
    prompt_version: str
    videos: tuple[VideoDescriptionRecord, ...]
    errors: tuple[dict[str, Any], ...] = ()
    version: int = 1
    generator: str = "arata_joycaption_shots"
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "generator": self.generator,
            "generated_at": self.generated_at,
            "source_path": self.source_path,
            "model": {
                "id": self.model_id,
                "caption_max_tokens": self.caption_max_tokens,
            },
            "prompt_version": self.prompt_version,
            "parameters": self.parameters,
            "videos": [record.to_payload() for record in self.videos],
            "errors": list(self.errors),
        }


@dataclass(frozen=True)
class ExportedFile:
    label: str
    file_path: str
    relative_output_path: str
    filename: str

    def to_ui_payload(self) -> dict[str, str]:
        return {
            "label": self.label,
            "filename": self.filename,
            "file_path": self.file_path,
            "relative_output_path": self.relative_output_path,
        }


@dataclass(frozen=True)
class JoyCaptionJsonExportResult:
    descriptions_file: ExportedFile


def _coerce_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
