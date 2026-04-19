try:
    from .analyze_shots import ArataJoyCaptionShotAnalyze
    from .export_json import ArataJoyCaptionShotJsonExport
except ImportError:
    if __package__ != "nodes":
        raise
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
else:
    NODE_CLASS_MAPPINGS = {
        "ArataJoyCaptionShotAnalyze": ArataJoyCaptionShotAnalyze,
        "ArataJoyCaptionShotJsonExport": ArataJoyCaptionShotJsonExport,
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
        "ArataJoyCaptionShotAnalyze": "Arata Analyze Shots (JoyCaption Beta One)",
        "ArataJoyCaptionShotJsonExport": "Arata Export JoyCaption Shot JSON",
    }

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
