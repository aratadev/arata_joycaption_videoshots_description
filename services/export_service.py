from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from ..utils.comfy_paths import get_output_directory
from ..utils.models import ExportedFile, JoyCaptionJsonExportResult, JoyCaptionShotDescriptionResult
from ..utils.path_utils import (
    build_output_filename_stem,
    ensure_directory,
    make_unique_path,
    normalize_output_subdirectory,
)


class JoyCaptionShotJsonExportService:
    def __init__(self, output_root: Path | None = None) -> None:
        self._output_root = Path(output_root) if output_root is not None else None

    def export(
        self,
        shot_descriptions: JoyCaptionShotDescriptionResult,
        output_subdirectory: str,
        filename_stem: str,
        overwrite_existing: bool,
    ) -> JoyCaptionJsonExportResult:
        output_root = self._output_root or get_output_directory()
        export_dir = ensure_directory(output_root / normalize_output_subdirectory(output_subdirectory))
        stem = build_output_filename_stem(shot_descriptions.source_path, filename_stem, fallback="joycaption_shots")

        descriptions_path = make_unique_path(export_dir / f"{stem}_joycaption_shots.json", overwrite_existing)
        descriptions_path.write_text(self._format_description_file(shot_descriptions), encoding="utf-8")

        return JoyCaptionJsonExportResult(
            descriptions_file=self._build_exported_file("descriptions", descriptions_path, output_root),
        )

    def _build_exported_file(self, label: str, file_path: Path, output_root: Path) -> ExportedFile:
        relative_output_path = file_path.resolve(strict=False).relative_to(output_root.resolve(strict=False)).as_posix()
        return ExportedFile(
            label=label,
            file_path=str(file_path),
            relative_output_path=relative_output_path,
            filename=file_path.name,
        )

    def _format_description_file(self, shot_descriptions: JoyCaptionShotDescriptionResult) -> str:
        payload = shot_descriptions.to_payload()
        payload["exported_at"] = datetime.now(timezone.utc).isoformat()
        return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
