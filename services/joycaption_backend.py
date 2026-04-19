from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..utils.models import SampledKeyframe, VideoMetadata
from ..utils.video_sampling import VideoFrameSample

DESCRIPTION_FIELDS = (
    "summary",
    "actions",
    "objects",
    "environment",
    "atmosphere",
    "shot_scale",
    "camera_motion",
    "temporal_notes",
)

LIST_DESCRIPTION_FIELDS = {"actions", "objects", "temporal_notes"}

JOYCAPTION_MODEL_ID = "fancyfeast/llama-joycaption-beta-one-hf-llava"

JOYCAPTION_INSTALL_HINT = (
    'Install it with: export COMFYUI_DIR="/path/to/ComfyUI"; '
    'export MODEL_DIR="$COMFYUI_DIR/models/joycaption/fancyfeast-llama-joycaption-beta-one-hf-llava"; '
    'mkdir -p "$MODEL_DIR"; '
    "huggingface-cli download fancyfeast/llama-joycaption-beta-one-hf-llava "
    '--local-dir "$MODEL_DIR" --local-dir-use-symlinks False. '
    "If Hugging Face requires authorization, run `huggingface-cli login` first or see README.md."
)

JOYCAPTION_REQUIRED_FILES = (
    "chat_template.json",
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


@dataclass(frozen=True)
class FrameCaption:
    keyframe: SampledKeyframe
    caption: str


class JoyCaptionLocalBackend:
    def __init__(
        self,
        model_id: str,
        model_path: Path,
        device: str,
        caption_max_tokens: int,
        prompt_version: str,
    ) -> None:
        self.model_id = str(model_id or JOYCAPTION_MODEL_ID)
        self.model_path = Path(model_path)
        self.device = str(device or "auto").lower()
        self.caption_max_tokens = int(caption_max_tokens)
        self.prompt_version = prompt_version
        self._processor: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None
        self._runtime_device: str = "cpu"
        self._model_dtype: Any | None = None

    def describe_video(
        self,
        video_path: Path,
        metadata: VideoMetadata,
        frame_samples: tuple[VideoFrameSample, ...],
        output_language: str,
    ) -> dict[str, Any] | str:
        captions = tuple(
            FrameCaption(
                keyframe=sample.keyframe,
                caption=self._caption_frame(sample.image, sample.keyframe, output_language),
            )
            for sample in frame_samples
        )
        try:
            raw_output = self._synthesize_shot_description(video_path, metadata, captions, output_language)
            parsed = self._parse_json_text(raw_output)
            if self._has_required_description_fields(parsed):
                return parsed
        except Exception:
            pass
        return self._build_fallback_description(captions)

    def repair_description(
        self,
        invalid_output: Any,
        error: Exception,
        output_language: str,
    ) -> dict[str, Any] | str:
        try:
            prompt = (
                "Fix the following output into one valid JSON object only. "
                "Do not add Markdown or commentary. Preserve the meaning. "
                f"Text values must be in {output_language}. Required keys: "
                "summary, actions, objects, environment, atmosphere, shot_scale, camera_motion, temporal_notes. "
                "actions, objects, and temporal_notes must be arrays of strings. "
                f"Parser error: {error}\n\nOutput:\n{invalid_output}"
            )
            raw_output = self._generate_text_only(prompt, max_new_tokens=1024)
            parsed = self._parse_json_text(raw_output)
            if self._has_required_description_fields(parsed):
                return parsed
        except Exception:
            pass
        return self._build_text_fallback_description(invalid_output)

    def _caption_frame(self, image: Any, keyframe: SampledKeyframe, output_language: str) -> str:
        prompt = (
            f"Write a long detailed description for this video frame in {output_language}. "
            "Begin with the main subject and visible action. Mention pivotal people, objects, scenery, "
            "spatial relationships, lighting, camera angle, shot scale, composition, and any visible motion cues. "
            "Use concrete details and avoid saying what is absent."
        )
        caption = self._generate_with_image(prompt, image, max_new_tokens=self.caption_max_tokens).strip()
        if caption:
            return caption
        timestamp = f" at {keyframe.timestamp_sec:.3f}s" if keyframe.timestamp_sec is not None else ""
        return f"Frame {keyframe.index}{timestamp} could not be captioned."

    def _synthesize_shot_description(
        self,
        video_path: Path,
        metadata: VideoMetadata,
        captions: tuple[FrameCaption, ...],
        output_language: str,
    ) -> str:
        timeline = "\n".join(
            f"- frame {item.keyframe.index}: source_frame={item.keyframe.frame_index}, "
            f"timestamp_sec={item.keyframe.timestamp_sec}, caption={json.dumps(item.caption, ensure_ascii=False)}"
            for item in captions
        )
        prompt = f"""
You are turning frame-by-frame image captions into one structured video shot description.

Source file: {video_path.name}
Duration seconds: {metadata.duration_sec}
FPS: {metadata.fps}
Total frames: {metadata.total_frames}

Frame captions in temporal order:
{timeline}

Return exactly one JSON object and no Markdown. Text values must be in {output_language}.
Required keys:
- summary: one concise shot-level summary.
- actions: array of visible actions or changes over time.
- objects: array of important visible people, objects, and scene elements.
- environment: description of the location or setting.
- atmosphere: visual mood, lighting, color, or tone.
- shot_scale: likely shot scale or framing.
- camera_motion: visible camera movement, or "static/unclear" if not inferable.
- temporal_notes: array of timestamped changes across the sampled frames.
""".strip()
        return self._generate_text_only(prompt, max_new_tokens=1024)

    def _ensure_loaded(self) -> None:
        if self._processor is not None and self._model is not None:
            return

        self._validate_local_model_path()

        try:
            import torch
            from transformers import AutoProcessor, LlavaForConditionalGeneration
        except Exception as exc:
            raise RuntimeError(
                "JoyCaption local inference requires `torch`, `transformers`, `accelerate`, and `pillow`. "
                "Install the package requirements and restart ComfyUI."
            ) from exc

        self._torch = torch
        self._runtime_device = self._resolve_runtime_device(torch)
        self._model_dtype = self._select_model_dtype(torch, self._runtime_device)

        model_path_text = str(self.model_path)
        self._processor = AutoProcessor.from_pretrained(model_path_text, local_files_only=True)

        model_kwargs: dict[str, Any] = {
            "local_files_only": True,
            "torch_dtype": self._model_dtype,
        }
        if self._runtime_device == "cuda":
            model_kwargs["device_map"] = 0

        try:
            self._model = LlavaForConditionalGeneration.from_pretrained(model_path_text, **model_kwargs)
        except TypeError:
            model_kwargs["dtype"] = model_kwargs.pop("torch_dtype")
            self._model = LlavaForConditionalGeneration.from_pretrained(model_path_text, **model_kwargs)

        if self._runtime_device in {"cpu", "mps"} and hasattr(self._model, "to"):
            self._model.to(self._runtime_device)
        if hasattr(self._model, "eval"):
            self._model.eval()

    def _validate_local_model_path(self) -> None:
        if not self.model_path.is_dir():
            raise RuntimeError(
                f"JoyCaption model folder not found: {self.model_path}. "
                + JOYCAPTION_INSTALL_HINT
            )

        missing = [name for name in JOYCAPTION_REQUIRED_FILES if not (self.model_path / name).is_file()]
        if missing:
            raise RuntimeError(
                f"JoyCaption model folder is incomplete: {self.model_path}. "
                f"Missing files: {', '.join(missing)}. "
                + JOYCAPTION_INSTALL_HINT
            )

        has_single_file = (self.model_path / "model.safetensors").is_file()
        shard_paths = sorted(self.model_path.glob("model-*.safetensors"))
        has_sharded_files = (self.model_path / "model.safetensors.index.json").is_file() and bool(shard_paths)
        if not has_single_file and not has_sharded_files:
            raise RuntimeError(
                f"JoyCaption model folder is incomplete: {self.model_path}. "
                "Expected model.safetensors or model.safetensors.index.json with model-*.safetensors shards. "
                + JOYCAPTION_INSTALL_HINT
            )

    def _resolve_runtime_device(self, torch: Any) -> str:
        cuda_available = bool(torch.cuda.is_available())
        mps_backend = getattr(torch.backends, "mps", None)
        mps_available = bool(mps_backend is not None and mps_backend.is_available())

        if self.device == "auto":
            if cuda_available:
                return "cuda"
            if mps_available:
                return "mps"
            return "cpu"
        if self.device == "cuda" and not cuda_available:
            raise RuntimeError("JoyCaption was configured for CUDA, but CUDA is not available.")
        if self.device == "mps" and not mps_available:
            raise RuntimeError("JoyCaption was configured for MPS, but MPS is not available.")
        if self.device not in {"cuda", "mps", "cpu"}:
            raise RuntimeError(f"Unsupported JoyCaption device: {self.device}")
        return self.device

    def _select_model_dtype(self, torch: Any, runtime_device: str) -> Any:
        if runtime_device == "cuda":
            if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        if runtime_device == "mps":
            return torch.float16
        return torch.float32

    def _generate_with_image(self, prompt: str, image: Any, max_new_tokens: int) -> str:
        self._ensure_loaded()
        processor = self._processor
        model = self._model
        if processor is None or model is None:
            raise RuntimeError("JoyCaption backend was not initialized.")

        torch = self._torch
        if torch is None:
            raise RuntimeError("JoyCaption torch runtime was not initialized.")

        pil_image = self._to_pil_rgb(image)
        conversation = [
            {"role": "system", "content": "You are a helpful image captioner."},
            {"role": "user", "content": prompt},
        ]
        conversation_text = processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = processor(text=[conversation_text], images=[pil_image], return_tensors="pt")
        inputs = self._move_inputs_to_device(inputs)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self._model_dtype)

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                suppress_tokens=None,
                use_cache=True,
            )[0]
        generated = generated[inputs["input_ids"].shape[1] :]
        return self._decode_generated_tokens(generated)

    def _generate_text_only(self, prompt: str, max_new_tokens: int) -> str:
        self._ensure_loaded()
        processor = self._processor
        model = self._model
        if processor is None or model is None:
            raise RuntimeError("JoyCaption backend was not initialized.")

        torch = self._torch
        if torch is None:
            raise RuntimeError("JoyCaption torch runtime was not initialized.")

        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("JoyCaption processor does not expose a tokenizer for text-only synthesis.")

        conversation = [
            {"role": "system", "content": "Return strict JSON only."},
            {"role": "user", "content": prompt},
        ]
        conversation_text = processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer([conversation_text], return_tensors="pt")
        inputs = self._move_inputs_to_device(inputs)

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                suppress_tokens=None,
                use_cache=True,
            )[0]
        generated = generated[inputs["input_ids"].shape[1] :]
        return self._decode_generated_tokens(generated)

    def _move_inputs_to_device(self, inputs: Any) -> Any:
        device = self._input_device()
        if hasattr(inputs, "to"):
            return inputs.to(device)
        for key, value in list(inputs.items()):
            if hasattr(value, "to"):
                inputs[key] = value.to(device)
        return inputs

    def _input_device(self) -> Any:
        model = self._model
        if model is not None and hasattr(model, "device"):
            return model.device
        return self._runtime_device

    def _to_pil_rgb(self, image: Any) -> Any:
        try:
            from PIL import Image
        except Exception as exc:
            raise RuntimeError("JoyCaption image conversion requires `pillow`.") from exc

        if isinstance(image, Image.Image):
            return image.convert("RGB")

        try:
            import numpy as np
        except Exception as exc:
            raise RuntimeError("JoyCaption frame conversion requires `numpy`.") from exc

        array = np.asarray(image)
        if array.ndim == 2:
            return Image.fromarray(array).convert("RGB")
        if array.ndim == 3 and array.shape[-1] in {3, 4}:
            return Image.fromarray(array.astype("uint8", copy=False)).convert("RGB")
        raise TypeError(f"Unsupported frame image shape for JoyCaption: {array.shape}")

    def _decode_generated_tokens(self, generated_tokens: Any) -> str:
        processor = self._processor
        if processor is None:
            return ""
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is not None and hasattr(tokenizer, "decode"):
            return tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()
        return processor.decode(generated_tokens, skip_special_tokens=True).strip()

    def _build_fallback_description(self, captions: tuple[FrameCaption, ...]) -> dict[str, Any]:
        temporal_notes = [
            self._format_temporal_note(item.keyframe, item.caption)
            for item in captions
            if item.caption.strip()
        ]
        caption_texts = [item.caption.strip() for item in captions if item.caption.strip()]
        summary = self._join_limited(caption_texts[:3], max_chars=900)
        if not summary:
            summary = "No reliable frame caption was generated."

        return {
            "summary": summary,
            "actions": self._caption_list(caption_texts, fallback=summary),
            "objects": self._caption_list(caption_texts, fallback=summary),
            "environment": caption_texts[0] if caption_texts else summary,
            "atmosphere": caption_texts[-1] if caption_texts else summary,
            "shot_scale": "Inferred from frame captions; review manually.",
            "camera_motion": "static/unclear from per-frame captions",
            "temporal_notes": temporal_notes or [summary],
        }

    def _build_text_fallback_description(self, invalid_output: Any) -> dict[str, Any]:
        text = str(invalid_output or "").strip() or "No reliable model output was generated."
        text = self._strip_code_fence(text)
        text = text[:1200].strip() or "No reliable model output was generated."
        return {
            "summary": text,
            "actions": [text],
            "objects": [text],
            "environment": text,
            "atmosphere": text,
            "shot_scale": "unclear",
            "camera_motion": "unclear",
            "temporal_notes": [text],
        }

    def _caption_list(self, captions: list[str], fallback: str) -> list[str]:
        values = []
        seen: set[str] = set()
        for caption in captions:
            cleaned = caption.strip()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            values.append(self._limit_text(cleaned, 260))
            if len(values) >= 8:
                break
        return values or [fallback]

    def _format_temporal_note(self, keyframe: SampledKeyframe, caption: str) -> str:
        if keyframe.timestamp_sec is None:
            prefix = f"Frame {keyframe.index}"
        else:
            prefix = f"{keyframe.timestamp_sec:.3f}s"
        return f"{prefix}: {self._limit_text(caption, 420)}"

    def _join_limited(self, values: list[str], max_chars: int) -> str:
        text = " ".join(value.strip() for value in values if value.strip())
        return self._limit_text(text, max_chars)

    def _limit_text(self, text: str, max_chars: int) -> str:
        cleaned = " ".join(str(text or "").split())
        if len(cleaned) <= max_chars:
            return cleaned
        return cleaned[: max(0, max_chars - 1)].rstrip() + "..."

    def _has_required_description_fields(self, value: Any) -> bool:
        if not isinstance(value, dict):
            return False
        return all(field in value for field in DESCRIPTION_FIELDS)

    def _parse_json_text(self, text: str) -> dict[str, Any]:
        cleaned = self._strip_code_fence(str(text or "").strip())
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            cleaned = cleaned[start : end + 1]
        parsed = json.loads(cleaned)
        if not isinstance(parsed, dict):
            raise TypeError("JoyCaption output must parse to a JSON object.")
        return parsed

    def _strip_code_fence(self, text: str) -> str:
        cleaned = str(text or "").strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`").strip()
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()
        return cleaned
