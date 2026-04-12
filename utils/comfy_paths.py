from __future__ import annotations

from pathlib import Path

try:
    import folder_paths
except Exception:
    folder_paths = None

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def get_input_directory() -> Path:
    if folder_paths is not None and hasattr(folder_paths, "get_input_directory"):
        return Path(folder_paths.get_input_directory())
    return _PACKAGE_ROOT / "input"


def get_output_directory() -> Path:
    if folder_paths is not None and hasattr(folder_paths, "get_output_directory"):
        return Path(folder_paths.get_output_directory())
    return _PACKAGE_ROOT / "output"


def get_annotated_filepath(path_text: str) -> str | None:
    if folder_paths is None or not hasattr(folder_paths, "get_annotated_filepath"):
        return None
    try:
        resolved = folder_paths.get_annotated_filepath(path_text)
    except Exception:
        return None
    return str(resolved) if resolved else None
