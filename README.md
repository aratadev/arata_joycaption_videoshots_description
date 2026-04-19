# Arata JoyCaption Video Shot Descriptions

ComfyUI custom nodes for analyzing short video shots with local JoyCaption Beta One and exporting one consolidated JSON description file.

## Nodes

- `Arata Analyze Shots (JoyCaption Beta One)`
- `Arata Export JoyCaption Shot JSON`

## Workflow Shape

1. Put video shots in ComfyUI input, or provide an absolute path to a single video file or a folder of video files.
2. Feed `source_path` into `Arata Analyze Shots (JoyCaption Beta One)`.
3. Connect `shot_descriptions` into `Arata Export JoyCaption Shot JSON`.
4. Run the workflow and use the export node download button to fetch the generated `.json` file.

Folder processing is flat and sorted by filename. Nested folders are not traversed.

## JoyCaption Beta One Backend

The node uses local Hugging Face Transformers inference. The default model id is `fancyfeast/llama-joycaption-beta-one-hf-llava`, and the default local model path is `joycaption/fancyfeast-llama-joycaption-beta-one-hf-llava` under `ComfyUI/models`. Absolute `model_path` values are used as-is; relative values are resolved from `ComfyUI/models`. Outside ComfyUI, relative paths resolve from this package's local `models` folder.

The backend loads with `local_files_only=True`, so it will not silently download the model during analysis. Download the files before running the node:

```bash
export COMFYUI_DIR="/path/to/ComfyUI"
export MODEL_DIR="$COMFYUI_DIR/models/joycaption/fancyfeast-llama-joycaption-beta-one-hf-llava"

# If Hugging Face requires authorization:
# huggingface-cli login

mkdir -p "$MODEL_DIR"
huggingface-cli download fancyfeast/llama-joycaption-beta-one-hf-llava \
  --local-dir "$MODEL_DIR" \
  --local-dir-use-symlinks False
```

JoyCaption Beta One is a LLaVA-style single-image captioning model. The model card recommends `AutoProcessor` with `LlavaForConditionalGeneration`, and the upstream README warns that Beta One is not a fully reliable general instruction follower. This node keeps the existing video sampling architecture but runs JoyCaption per sampled frame, then tries a second text-only pass to synthesize the final shot JSON. If that pass is unsupported or returns invalid JSON, the node builds a deterministic fallback JSON from the frame captions.

References:

- [JoyCaption Beta One model card](https://huggingface.co/fancyfeast/llama-joycaption-beta-one-hf-llava)
- [JoyCaption README](https://github.com/fpgaminer/joycaption)

Video is processed as an ordered sequence of sampled frames:

- Up to 1 frame per second for shots around 60 seconds or shorter.
- A hard cap of 60 sampled frames.
- Longer clips are sampled evenly across the full duration and marked with a warning.

The default `caption_max_tokens` is `512`. Available values are `128`, `256`, `512`, `768`, and `1024`.

## Output Format

The export node writes a UTF-8 JSON file under `output/arata_joycaption_shots` by default:

```json
{
  "version": 1,
  "generator": "arata_joycaption_shots",
  "source_path": "/path/to/shots",
  "model": {
    "id": "fancyfeast/llama-joycaption-beta-one-hf-llava",
    "caption_max_tokens": 512
  },
  "videos": [
    {
      "source_video_path": "/path/to/shots/shot001.mp4",
      "filename": "shot001.mp4",
      "status": "completed",
      "metadata": {
        "fps": 24.0,
        "total_frames": 240,
        "duration_sec": 10.0
      },
      "sampled_keyframes": [
        {
          "index": 1,
          "frame_index": 0,
          "timestamp_sec": 0.0
        }
      ],
      "description": {
        "summary": "Краткое описание шота.",
        "actions": ["Основное действие."],
        "objects": ["Видимые объекты."],
        "environment": "Окружение.",
        "atmosphere": "Атмосфера.",
        "shot_scale": "Крупность плана.",
        "camera_motion": "Характер движения камеры.",
        "temporal_notes": ["Изменения во времени."]
      },
      "warnings": []
    }
  ],
  "errors": []
}
```

The node caches per-video results under the ComfyUI output cache. Re-running the same folder reuses completed cached records when the source file signature, model id, local model path signature, caption token limit, and prompt version match. Failed video records are reported in the current run but are not cached. Local model setup, dependency, device, and inference failures stop the workflow instead of being reported as damaged video files.

## Manual Smoke Test

After installing requirements and the local JoyCaption model folder, run ComfyUI, select a short shot, run the two-node workflow, and confirm that the export JSON contains Russian values under the expected fields.

Known limitation: the second synthesis pass uses JoyCaption as a text-only LLM even though the model is primarily trained for image captioning. The deterministic fallback is intentionally retained so batch processing remains stable if that pass is weak or unavailable.
