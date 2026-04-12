from __future__ import annotations

from ..services.export_service import GemmaShotJsonExportService


class ArataGemmaShotJsonExport:
    CATEGORY = "Arata/Video Analysis"
    FUNCTION = "export_file"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("descriptions_json_path",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "shot_descriptions": ("ARATA_GEMMA_SHOT_DESCRIPTIONS",),
                "output_subdirectory": ("STRING", {"default": "arata_gemma_shots"}),
                "filename_stem": ("STRING", {"default": ""}),
                "overwrite_existing": ("BOOLEAN", {"default": False}),
            }
        }

    def export_file(
        self,
        shot_descriptions,
        output_subdirectory: str,
        filename_stem: str,
        overwrite_existing: bool,
    ):
        service = GemmaShotJsonExportService()
        export_result = service.export(
            shot_descriptions=shot_descriptions,
            output_subdirectory=output_subdirectory,
            filename_stem=filename_stem,
            overwrite_existing=bool(overwrite_existing),
        )
        ui_payload = [export_result.descriptions_file.to_ui_payload()]
        return {
            "ui": {"description_files": ui_payload},
            "result": (export_result.descriptions_file.file_path,),
        }
