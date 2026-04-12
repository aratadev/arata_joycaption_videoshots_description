from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..utils.models import VideoMetadata
from ..utils.video_sampling import VideoFrameSample

SHOT_DESCRIPTION_TOOL = {
    "type": "function",
    "function": {
        "name": "save_shot_description",
        "description": "Save a structured description for one video shot.",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "actions": {"type": "array", "items": {"type": "string"}},
                "objects": {"type": "array", "items": {"type": "string"}},
                "environment": {"type": "string"},
                "atmosphere": {"type": "string"},
                "shot_scale": {"type": "string"},
                "camera_motion": {"type": "string"},
                "temporal_notes": {"type": "array", "items": {"type": "string"}},
            },
            "required": [
                "summary",
                "actions",
                "objects",
                "environment",
                "atmosphere",
                "shot_scale",
                "camera_motion",
                "temporal_notes",
            ],
            "additionalProperties": False,
        },
    },
}

GEMMA_4_WGET_INSTALL_HINT = (
    'Install it with: export COMFYUI_DIR="/path/to/ComfyUI"; '
    'export MODEL_DIR="$COMFYUI_DIR/models/gemma/google-gemma-4-E4B-it"; '
    'mkdir -p "$MODEL_DIR"; '
    'for f in chat_template.jinja config.json generation_config.json model.safetensors '
    'processor_config.json tokenizer.json tokenizer_config.json; do '
    'wget -c --show-progress -O "$MODEL_DIR/$f" '
    '"https://huggingface.co/google/gemma-4-E4B-it/resolve/main/$f?download=true"; done. '
    "If Hugging Face requires authorization, add an Authorization bearer header or see README.md."
)


class Gemma4LocalBackend:
    def __init__(
        self,
        model_id: str,
        model_path: Path,
        device: str,
        visual_token_budget: int,
        prompt_version: str,
    ) -> None:
        self.model_id = model_id
        self.model_path = Path(model_path)
        self.device = str(device or "auto").lower()
        self.visual_token_budget = int(visual_token_budget)
        self.prompt_version = prompt_version
        self._processor: Any | None = None
        self._model: Any | None = None

    def describe_video(
        self,
        video_path: Path,
        metadata: VideoMetadata,
        frame_samples: tuple[VideoFrameSample, ...],
        output_language: str,
    ) -> dict[str, Any] | str:
        raw_output = self._generate_description(video_path, metadata, frame_samples, output_language)
        try:
            return self._parse_json_text(raw_output)
        except Exception as exc:
            return self.repair_description(raw_output, exc, output_language)

    def repair_description(
        self,
        invalid_output: Any,
        error: Exception,
        output_language: str,
    ) -> dict[str, Any] | str:
        self._ensure_loaded()
        prompt = (
            "Fix the following model output into one valid JSON object only. "
            "Do not add Markdown or commentary. Preserve the meaning. "
            f"Text values must be in {output_language}. Required keys: "
            "summary, actions, objects, environment, atmosphere, shot_scale, camera_motion, temporal_notes. "
            "actions, objects, and temporal_notes must be arrays of strings. "
            f"Parser error: {error}\n\nOutput:\n{invalid_output}"
        )
        raw = self._generate_from_messages(
            [
                {"role": "system", "content": [{"type": "text", "text": "Return strict JSON only."}]},
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
            ],
            max_new_tokens=1536,
        )
        return self._parse_json_text(raw)

    def _generate_description(
        self,
        video_path: Path,
        metadata: VideoMetadata,
        frame_samples: tuple[VideoFrameSample, ...],
        output_language: str,
    ) -> str:
        timeline = "\n".join(
            f"- frame {sample.keyframe.index}: source_frame={sample.keyframe.frame_index}, "
            f"timestamp_sec={sample.keyframe.timestamp_sec}"
            for sample in frame_samples
        )
        user_prompt = f"""
Analyze this short video shot as a temporal sequence of sampled frames.

Source file: {video_path.name}
Duration seconds: {metadata.duration_sec}
FPS: {metadata.fps}
Total frames: {metadata.total_frames}
Frame timeline, in the same order as the images:
{timeline}

Use the `save_shot_description` tool with the required fields. If tool calling is unavailable,
return exactly one JSON object with the same fields and no Markdown. All descriptive text values
must be in {output_language}.
""".strip()

        content = [{"type": "image", "image": sample.image} for sample in frame_samples]
        content.append({"type": "text", "text": user_prompt})
        return self._generate_from_messages(
            [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "You describe film/video shots for downstream JSON datasets. "
                                "Use tool calls when available; otherwise answer only with valid JSON "
                                "matching the requested schema."
                            ),
                        }
                    ],
                },
                {"role": "user", "content": content},
            ],
            max_new_tokens=2048,
            tools=[SHOT_DESCRIPTION_TOOL],
        )

    def _ensure_loaded(self) -> None:
        if self._processor is not None and self._model is not None:
            return

        self._validate_local_model_path()

        try:
            import torch
            from transformers import AutoProcessor
        except Exception as exc:
            raise RuntimeError(
                "Gemma 4 local inference requires `torch`, `transformers`, and their model dependencies. "
                "Install the package requirements and restart ComfyUI."
            ) from exc

        model_cls = self._resolve_multimodal_model_class()

        dtype = torch.float32
        if torch.cuda.is_available() or getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            dtype = torch.bfloat16

        model_path_text = str(self.model_path)
        self._processor = AutoProcessor.from_pretrained(
            model_path_text,
            padding_side="left",
            local_files_only=True,
        )
        self._set_processor_visual_budget(self._processor)
        model_kwargs: dict[str, Any] = {"dtype": dtype}
        if self.device in {"auto", "cuda"}:
            model_kwargs["device_map"] = "auto"

        self._model = model_cls.from_pretrained(model_path_text, local_files_only=True, **model_kwargs)
        if self.device in {"cpu", "mps"} and hasattr(self._model, "to"):
            self._model.to(self.device)
        if hasattr(self._model, "eval"):
            self._model.eval()

    def _validate_local_model_path(self) -> None:
        required_files = (
            "chat_template.jinja",
            "config.json",
            "model.safetensors",
            "processor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
        )
        if not self.model_path.is_dir():
            raise RuntimeError(
                f"Gemma 4 model folder not found: {self.model_path}. "
                + GEMMA_4_WGET_INSTALL_HINT
            )

        missing = [name for name in required_files if not (self.model_path / name).is_file()]
        if missing:
            raise RuntimeError(
                f"Gemma 4 model folder is incomplete: {self.model_path}. "
                f"Missing files: {', '.join(missing)}. "
                + GEMMA_4_WGET_INSTALL_HINT
            )

    def _set_processor_visual_budget(self, processor: Any) -> None:
        for component_name in ("image_processor", "video_processor"):
            component = getattr(processor, component_name, None)
            if component is not None and hasattr(component, "max_soft_tokens"):
                setattr(component, "max_soft_tokens", self.visual_token_budget)

    def _resolve_multimodal_model_class(self):
        errors: list[str] = []
        for class_name in (
            "AutoModelForMultimodalLM",
            "AutoModelForImageTextToText",
            "Gemma4ForConditionalGeneration",
        ):
            try:
                import transformers

                model_cls = getattr(transformers, class_name)
                return model_cls
            except Exception as exc:
                errors.append(f"{class_name}: {exc}")

        raise RuntimeError(
            "This Transformers installation does not expose a compatible Gemma 4 multimodal model class. "
            "Upgrade `transformers`. Tried: " + "; ".join(errors)
        )

    def _generate_from_messages(
        self,
        messages: list[dict[str, Any]],
        max_new_tokens: int,
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        self._ensure_loaded()
        processor = self._processor
        model = self._model
        if processor is None or model is None:
            raise RuntimeError("Gemma 4 backend was not initialized.")

        inputs = self._apply_chat_template(processor, messages, tools)

        if hasattr(inputs, "to") and hasattr(model, "device"):
            inputs = inputs.to(model.device)

        input_len = inputs["input_ids"].shape[-1]
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        generated_tokens = outputs[0][input_len:]
        response = processor.decode(generated_tokens, skip_special_tokens=False)
        tool_arguments = self._extract_tool_arguments(processor, response)
        if tool_arguments is not None:
            return json.dumps(tool_arguments, ensure_ascii=False)
        return response

    def _apply_chat_template(
        self,
        processor: Any,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "tokenize": True,
            "return_dict": True,
            "return_tensors": "pt",
            "add_generation_prompt": True,
            "max_soft_tokens": self.visual_token_budget,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "save_shot_description"

        try:
            return processor.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("tool_choice", None)
            try:
                return processor.apply_chat_template(messages, **kwargs)
            except TypeError:
                kwargs.pop("tools", None)
                try:
                    return processor.apply_chat_template(messages, **kwargs)
                except TypeError:
                    kwargs.pop("max_soft_tokens", None)
                    return processor.apply_chat_template(messages, **kwargs)

    def _extract_tool_arguments(self, processor: Any, response: str) -> dict[str, Any] | None:
        parsed_response = self._parse_processor_response(processor, response)
        tool_arguments = self._find_tool_arguments(parsed_response)
        if tool_arguments is not None:
            return tool_arguments
        return self._find_tool_arguments(self._parse_json_object(response))

    def _parse_processor_response(self, processor: Any, response: str) -> Any:
        parse_response = getattr(processor, "parse_response", None)
        if not callable(parse_response):
            return None
        try:
            return parse_response(response)
        except Exception:
            return None

    def _find_tool_arguments(self, value: Any) -> dict[str, Any] | None:
        if isinstance(value, dict):
            function = value.get("function")
            name = value.get("name")
            if name is None and isinstance(function, dict):
                name = function.get("name")

            arguments = value.get("arguments", value.get("args"))
            if arguments is None and isinstance(function, dict):
                arguments = function.get("arguments", function.get("args"))
            if name == "save_shot_description" and arguments is not None:
                return self._coerce_tool_arguments(arguments)

            for key in ("tool_calls", "tool_call", "function_call", "calls"):
                nested = value.get(key)
                found = self._find_tool_arguments(nested)
                if found is not None:
                    return found
        if isinstance(value, list):
            for item in value:
                found = self._find_tool_arguments(item)
                if found is not None:
                    return found
        return None

    def _coerce_tool_arguments(self, arguments: Any) -> dict[str, Any] | None:
        if isinstance(arguments, dict):
            return arguments
        if isinstance(arguments, str):
            return self._parse_json_object(arguments)
        return None

    def _parse_json_object(self, text: Any) -> dict[str, Any] | None:
        if not isinstance(text, str):
            return None
        cleaned = str(text or "").strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`").strip()
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            cleaned = cleaned[start : end + 1]
        try:
            parsed = json.loads(cleaned)
        except Exception:
            return None
        return parsed if isinstance(parsed, dict) else None

    def _parse_json_text(self, text: str) -> dict[str, Any]:
        cleaned = str(text or "").strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`").strip()
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            cleaned = cleaned[start : end + 1]
        parsed = json.loads(cleaned)
        if not isinstance(parsed, dict):
            raise TypeError("Gemma output must parse to a JSON object.")
        return parsed
