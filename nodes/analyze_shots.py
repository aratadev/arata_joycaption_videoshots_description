from __future__ import annotations

from ..services.joycaption_analysis_service import (
    DEFAULT_JOYCAPTION_MODEL_PATH,
    JOYCAPTION_MODEL_IDS,
    PROMPT_VERSION,
    JoyCaptionShotAnalysisService,
)
from ..utils.path_utils import build_model_path_signature, build_source_signature


class ArataJoyCaptionShotAnalyze:
    CATEGORY = "Arata/Video Analysis"
    FUNCTION = "analyze_shots"
    RETURN_TYPES = ("ARATA_JOYCAPTION_SHOT_DESCRIPTIONS",)
    RETURN_NAMES = ("shot_descriptions",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_path": ("STRING", {"default": ""}),
                "model_id": (JOYCAPTION_MODEL_IDS, {"default": "fancyfeast/llama-joycaption-beta-one-hf-llava"}),
                "model_path": ("STRING", {"default": DEFAULT_JOYCAPTION_MODEL_PATH}),
                "device": (["auto", "cuda", "mps", "cpu"], {"default": "auto"}),
                "caption_max_tokens": (["128", "256", "512", "768", "1024"], {"default": "512"}),
            }
        }

    @classmethod
    def IS_CHANGED(
        cls,
        source_path: str,
        model_id: str = "",
        model_path: str = DEFAULT_JOYCAPTION_MODEL_PATH,
        caption_max_tokens: str = "512",
        **_: object,
    ) -> str:
        return ":".join(
            [
                build_source_signature(source_path),
                str(model_id or ""),
                build_model_path_signature(model_path or DEFAULT_JOYCAPTION_MODEL_PATH),
                str(caption_max_tokens or ""),
                PROMPT_VERSION,
            ]
        )

    def analyze_shots(
        self,
        source_path: str,
        model_id: str,
        model_path: str,
        device: str,
        caption_max_tokens: str,
    ):
        service = JoyCaptionShotAnalysisService()
        result = service.analyze(
            source_path=source_path,
            model_id=model_id,
            model_path=model_path,
            device=device,
            caption_max_tokens=int(caption_max_tokens),
        )
        return (result,)
