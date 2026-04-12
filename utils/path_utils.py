from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .comfy_paths import get_annotated_filepath, get_input_directory

VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".m4v",
    ".avi",
    ".mkv",
    ".webm",
    ".mpg",
    ".mpeg",
    ".wmv",
}

_INVALID_PATH_CHARS_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def strip_comfy_path_annotation(path_text: str) -> str:
    value = str(path_text or "").strip()
    if value.endswith("]") and " [" in value:
        value = value.rsplit(" [", 1)[0].strip()
    return value


def resolve_source_path(path_text: str) -> Path:
    candidate_text = strip_comfy_path_annotation(path_text)
    if not candidate_text:
        raise ValueError("Source path is empty.")

    candidate = Path(candidate_text).expanduser()
    if candidate.exists():
        return candidate.resolve()

    annotated = get_annotated_filepath(candidate_text)
    if annotated:
        annotated_path = Path(annotated)
        if annotated_path.exists():
            return annotated_path.resolve()

    input_directory = get_input_directory()
    for variant in (
        input_directory / candidate_text,
        input_directory / Path(candidate_text).name,
    ):
        if variant.exists():
            return variant.resolve()

    return candidate.resolve(strict=False) if candidate.is_absolute() else candidate


def discover_video_files(source_path: Path) -> list[Path]:
    candidate = Path(source_path).expanduser()
    if candidate.is_file():
        if _is_supported_video(candidate):
            return [candidate.resolve()]
        raise ValueError(f"Unsupported video file extension: {candidate}")

    if not candidate.is_dir():
        raise ValueError(f"Source path not found: {source_path}")

    video_paths = sorted(
        (path.resolve() for path in candidate.iterdir() if path.is_file() and _is_supported_video(path)),
        key=lambda path: path.name.lower(),
    )
    if not video_paths:
        raise ValueError(f"No supported video files found in folder: {source_path}")
    return video_paths


def build_file_signature(path_text: str) -> str:
    candidate = Path(path_text)
    if not str(path_text) or not candidate.is_file():
        return str(path_text)
    stat = candidate.stat()
    return f"{candidate.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"


def build_source_signature(path_text: str) -> str:
    try:
        resolved = resolve_source_path(path_text)
    except Exception:
        return str(path_text or "")

    if resolved.is_file():
        return build_file_signature(str(resolved))

    if resolved.is_dir():
        signatures = []
        for video_path in sorted(
            (path for path in resolved.iterdir() if path.is_file() and _is_supported_video(path)),
            key=lambda path: path.name.lower(),
        ):
            signatures.append({"name": video_path.name, "signature": build_file_signature(str(video_path))})
        return _sha256_json({"type": "dir", "path": str(resolved), "files": signatures})

    return str(resolved)


def build_resume_cache_key(
    file_signature: str,
    model_id: str,
    visual_token_budget: int,
    prompt_version: str,
) -> str:
    return _sha256_json(
        {
            "file_signature": file_signature,
            "model_id": model_id,
            "visual_token_budget": int(visual_token_budget),
            "prompt_version": prompt_version,
        }
    )


def normalize_output_subdirectory(subdirectory: str) -> Path:
    raw_value = str(subdirectory or "").strip().replace("\\", "/").strip("/")
    if not raw_value:
        return Path("arata_gemma_shots")

    parts: list[str] = []
    for raw_part in raw_value.split("/"):
        part = raw_part.strip()
        if not part or part == ".":
            continue
        if part == "..":
            raise ValueError("Output subdirectory must stay inside the ComfyUI output directory.")
        parts.append(sanitize_path_component(part))

    if not parts:
        return Path("arata_gemma_shots")
    return Path(*parts)


def build_output_filename_stem(source_path: str, filename_stem: str, fallback: str = "gemma_shots") -> str:
    explicit_stem = sanitize_filename_stem(filename_stem)
    if explicit_stem:
        return explicit_stem
    return sanitize_filename_stem(Path(source_path).stem) or fallback


def sanitize_filename_stem(value: str) -> str:
    text = _INVALID_PATH_CHARS_RE.sub("_", str(value or "").strip())
    text = re.sub(r"\s+", "_", text)
    return text.strip("._")


def sanitize_path_component(value: str) -> str:
    cleaned = _INVALID_PATH_CHARS_RE.sub("_", str(value or "").strip())
    cleaned = cleaned.strip(".")
    if not cleaned:
        raise ValueError("Encountered an empty output path component after sanitization.")
    return cleaned


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_unique_path(path: Path, overwrite_existing: bool) -> Path:
    if overwrite_existing or not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    index = 1
    while True:
        candidate = parent / f"{stem}_{index:03d}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def resolve_download_path(relative_path: str, output_root: Path) -> Path:
    cleaned = str(relative_path or "").strip().replace("\\", "/")
    if not cleaned:
        raise ValueError("Download path is empty.")

    candidate = Path(cleaned)
    if candidate.is_absolute():
        raise ValueError("Download path must be relative to the ComfyUI output directory.")

    resolved_root = output_root.resolve(strict=False)
    resolved_path = (resolved_root / candidate).resolve(strict=False)
    if resolved_root not in resolved_path.parents and resolved_path != resolved_root:
        raise ValueError("Download path escapes the ComfyUI output directory.")
    return resolved_path


def _is_supported_video(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTENSIONS


def _sha256_json(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
