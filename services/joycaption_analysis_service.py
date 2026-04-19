from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol

from .joycaption_backend import DESCRIPTION_FIELDS, JOYCAPTION_MODEL_ID, JoyCaptionLocalBackend
from ..utils.comfy_paths import get_output_directory
from ..utils.models import (
    JoyCaptionShotDescriptionResult,
    VideoDescriptionRecord,
    VideoMetadata,
)
from ..utils.path_utils import (
    build_file_signature,
    build_model_path_signature,
    build_resume_cache_key,
    discover_video_files,
    ensure_directory,
    resolve_model_path,
    resolve_source_path,
)
from ..utils.video_metadata import probe_video_metadata
from ..utils.video_sampling import VideoFrameSample, sample_video_frames

PROMPT_VERSION = "joycaption-shot-description-v1"
DEFAULT_JOYCAPTION_MODEL_ID = JOYCAPTION_MODEL_ID
DEFAULT_JOYCAPTION_MODEL_PATH = "joycaption/fancyfeast-llama-joycaption-beta-one-hf-llava"

JOYCAPTION_MODEL_IDS = [
    DEFAULT_JOYCAPTION_MODEL_ID,
]

LIST_DESCRIPTION_FIELDS = {"actions", "objects", "temporal_notes"}


class JoyCaptionDescriptionBackend(Protocol):
    def describe_video(
        self,
        video_path: Path,
        metadata: VideoMetadata,
        frame_samples: tuple[VideoFrameSample, ...],
        output_language: str,
    ) -> dict[str, Any] | str:
        ...

    def repair_description(
        self,
        invalid_output: Any,
        error: Exception,
        output_language: str,
    ) -> dict[str, Any] | str:
        ...


BackendFactory = Callable[[str, Path, str, int], JoyCaptionDescriptionBackend]
MetadataProbe = Callable[[str], VideoMetadata]
FrameSampler = Callable[[str, VideoMetadata], tuple[VideoFrameSample, ...]]


@dataclass(frozen=True)
class _ProcessContext:
    source_path: Path
    model_id: str
    model_path: Path
    model_path_signature: str
    device: str
    caption_max_tokens: int
    output_language: str


class JoyCaptionShotAnalysisService:
    def __init__(
        self,
        backend_factory: BackendFactory | None = None,
        metadata_probe: MetadataProbe = probe_video_metadata,
        frame_sampler: FrameSampler = sample_video_frames,
        cache_dir: Path | None = None,
    ) -> None:
        self._backend_factory = backend_factory or self._default_backend_factory
        self._metadata_probe = metadata_probe
        self._frame_sampler = frame_sampler
        self._cache = _ResumeCache(cache_dir or get_output_directory() / "cache" / "arata_joycaption_shots")

    def analyze(
        self,
        source_path: str,
        model_id: str,
        device: str,
        caption_max_tokens: int,
        output_language: str = "Russian",
        model_path: str = DEFAULT_JOYCAPTION_MODEL_PATH,
    ) -> JoyCaptionShotDescriptionResult:
        resolved_source = resolve_source_path(source_path)
        video_paths = discover_video_files(resolved_source)
        resolved_model_path = resolve_model_path(model_path or DEFAULT_JOYCAPTION_MODEL_PATH)
        context = _ProcessContext(
            source_path=resolved_source,
            model_id=str(model_id or DEFAULT_JOYCAPTION_MODEL_ID),
            model_path=resolved_model_path,
            model_path_signature=build_model_path_signature(str(resolved_model_path)),
            device=str(device or "auto"),
            caption_max_tokens=int(caption_max_tokens),
            output_language=output_language,
        )

        self._log(f"Processing {len(video_paths)} video file(s) from {resolved_source}")
        backend: JoyCaptionDescriptionBackend | None = None
        records: list[VideoDescriptionRecord] = []

        for index, video_path in enumerate(video_paths, start=1):
            self._log(f"[{index}/{len(video_paths)}] {video_path.name}")
            cache_key = build_resume_cache_key(
                file_signature=build_file_signature(str(video_path)),
                model_id=context.model_id,
                model_path_signature=context.model_path_signature,
                caption_max_tokens=context.caption_max_tokens,
                prompt_version=PROMPT_VERSION,
            )
            cached_record = self._cache.load(cache_key)
            if cached_record is not None:
                records.append(cached_record)
                self._log(f"Using cached result for {video_path.name}: {cached_record.status}")
                continue

            try:
                metadata = self._metadata_probe(str(video_path))
                frame_samples = self._frame_sampler(str(video_path), metadata)
                if not frame_samples:
                    raise RuntimeError("No keyframes could be sampled from the video.")
            except Exception as exc:
                self._log(f"Skipping {video_path.name}: {exc}")
                records.append(self._build_failed_record(video_path, cache_key, exc))
                continue

            if backend is None:
                backend = self._backend_factory(
                    context.model_id,
                    context.model_path,
                    context.device,
                    context.caption_max_tokens,
                )

            raw_description = backend.describe_video(
                video_path=video_path,
                metadata=metadata,
                frame_samples=frame_samples,
                output_language=context.output_language,
            )
            try:
                description = self._validate_description(raw_description)
            except Exception as validation_error:
                repaired_description = backend.repair_description(
                    invalid_output=raw_description,
                    error=validation_error,
                    output_language=context.output_language,
                )
                description = self._validate_description(repaired_description)

            warnings = self._build_video_warnings(metadata, frame_samples)
            record = VideoDescriptionRecord(
                source_video_path=str(video_path),
                filename=video_path.name,
                status="completed",
                metadata=metadata,
                keyframes=tuple(sample.keyframe for sample in frame_samples),
                description=description,
                warnings=tuple(warnings),
                cache_key=cache_key,
            )
            self._cache.save(cache_key, record)
            records.append(record)

        errors = tuple(record.to_error_payload() for record in records if record.status != "completed")
        return JoyCaptionShotDescriptionResult(
            source_path=str(resolved_source),
            generated_at=datetime.now(timezone.utc).isoformat(),
            model_id=context.model_id,
            caption_max_tokens=context.caption_max_tokens,
            prompt_version=PROMPT_VERSION,
            videos=tuple(records),
            errors=errors,
            parameters={
                "device": context.device,
                "output_language": context.output_language,
                "folder_traversal": "flat_sorted",
                "model_path": str(context.model_path),
                "frame_captioning": "per_keyframe",
                "shot_synthesis": "joycaption_text_pass_with_deterministic_fallback",
            },
        )

    def _default_backend_factory(
        self,
        model_id: str,
        model_path: Path,
        device: str,
        caption_max_tokens: int,
    ) -> JoyCaptionDescriptionBackend:
        return JoyCaptionLocalBackend(
            model_id=model_id,
            model_path=model_path,
            device=device,
            caption_max_tokens=caption_max_tokens,
            prompt_version=PROMPT_VERSION,
        )

    def _build_video_warnings(
        self,
        metadata: VideoMetadata,
        frame_samples: tuple[VideoFrameSample, ...],
    ) -> list[str]:
        warnings: list[str] = []
        if metadata.duration_sec and metadata.duration_sec > 60:
            warnings.append(
                "Video is longer than the recommended ~60 seconds for dense per-keyframe JoyCaption analysis; "
                "60 frames were sampled evenly across the full duration."
            )
        if len(frame_samples) >= 60:
            warnings.append("Frame sampling reached the 60-frame cap.")
        return warnings

    def _build_failed_record(self, video_path: Path, cache_key: str, exc: Exception) -> VideoDescriptionRecord:
        return VideoDescriptionRecord(
            source_video_path=str(video_path),
            filename=video_path.name,
            status="failed",
            warnings=("Processing skipped; the rest of the batch continued.",),
            error=str(exc),
            cache_key=cache_key,
        )

    def _validate_description(self, raw_description: dict[str, Any] | str) -> dict[str, Any]:
        data = self._coerce_description_json(raw_description)
        missing = [field for field in DESCRIPTION_FIELDS if field not in data]
        if missing:
            raise ValueError(f"JoyCaption response is missing required description fields: {', '.join(missing)}")

        normalized: dict[str, Any] = {}
        for field in DESCRIPTION_FIELDS:
            value = data[field]
            if field in LIST_DESCRIPTION_FIELDS:
                normalized[field] = self._coerce_string_list(value, field)
            else:
                normalized[field] = self._coerce_string(value, field)
        return normalized

    def _coerce_description_json(self, raw_description: dict[str, Any] | str) -> dict[str, Any]:
        if isinstance(raw_description, dict):
            return raw_description
        if not isinstance(raw_description, str):
            raise TypeError(f"JoyCaption response must be a JSON object or string, got {type(raw_description).__name__}.")
        text = raw_description.strip()
        if text.startswith("```"):
            text = text.strip("`").strip()
            if text.lower().startswith("json"):
                text = text[4:].strip()
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise TypeError("JoyCaption response JSON must be an object.")
        return parsed

    def _coerce_string_list(self, value: Any, field: str) -> list[str]:
        if isinstance(value, str):
            return [value]
        if not isinstance(value, list):
            raise TypeError(f"Description field '{field}' must be a list of strings.")
        return [self._coerce_string(item, field) for item in value if str(item).strip()]

    def _coerce_string(self, value: Any, field: str) -> str:
        if isinstance(value, list):
            return ", ".join(str(item).strip() for item in value if str(item).strip())
        if value is None:
            raise TypeError(f"Description field '{field}' must not be null.")
        return str(value).strip()

    def _log(self, message: str) -> None:
        print(f"[Arata JoyCaption Shots] {message}")


class _ResumeCache:
    def __init__(self, cache_dir: Path) -> None:
        self._cache_dir = cache_dir

    def load(self, cache_key: str) -> VideoDescriptionRecord | None:
        cache_path = self._cache_path(cache_key)
        if not cache_path.is_file():
            return None
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            record = VideoDescriptionRecord.from_payload(payload)
            return record if record.status == "completed" else None
        except Exception:
            return None

    def save(self, cache_key: str, record: VideoDescriptionRecord) -> None:
        ensure_directory(self._cache_dir)
        cache_path = self._cache_path(cache_key)
        cache_path.write_text(json.dumps(record.to_payload(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def _cache_path(self, cache_key: str) -> Path:
        return self._cache_dir / f"{cache_key}.json"
