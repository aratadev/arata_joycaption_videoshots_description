from .analyze_shots import ArataGemmaShotAnalyze
from .export_json import ArataGemmaShotJsonExport

NODE_CLASS_MAPPINGS = {
    "ArataGemmaShotAnalyze": ArataGemmaShotAnalyze,
    "ArataGemmaShotJsonExport": ArataGemmaShotJsonExport,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ArataGemmaShotAnalyze": "Arata Analyze Shots (Gemma 4)",
    "ArataGemmaShotJsonExport": "Arata Export Gemma Shot JSON",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
