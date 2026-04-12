from __future__ import annotations

from ..services.gemma_analysis_service import (
    GEMMA_4_MODEL_IDS,
    PROMPT_VERSION,
    GemmaShotAnalysisService,
)
from ..utils.path_utils import build_source_signature


class ArataGemmaShotAnalyze:
    CATEGORY = "Arata/Video Analysis"
    FUNCTION = "analyze_shots"
    RETURN_TYPES = ("ARATA_GEMMA_SHOT_DESCRIPTIONS",)
    RETURN_NAMES = ("shot_descriptions",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_path": ("STRING", {"default": ""}),
                "model_id": (GEMMA_4_MODEL_IDS, {"default": "google/gemma-4-E4B-it"}),
                "device": (["auto", "cuda", "mps", "cpu"], {"default": "auto"}),
                "visual_token_budget": (["70", "140", "280", "560", "1120"], {"default": "140"}),
            }
        }

    @classmethod
    def IS_CHANGED(
        cls,
        source_path: str,
        model_id: str = "",
        visual_token_budget: str = "140",
        **_: object,
    ) -> str:
        return ":".join(
            [
                build_source_signature(source_path),
                str(model_id or ""),
                str(visual_token_budget or ""),
                PROMPT_VERSION,
            ]
        )

    def analyze_shots(
        self,
        source_path: str,
        model_id: str,
        device: str,
        visual_token_budget: str,
    ):
        service = GemmaShotAnalysisService()
        result = service.analyze(
            source_path=source_path,
            model_id=model_id,
            device=device,
            visual_token_budget=int(visual_token_budget),
        )
        return (result,)
