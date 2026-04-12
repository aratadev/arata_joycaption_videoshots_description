# Arata Gemma Video Shot Descriptions

ComfyUI custom nodes for analyzing short video shots with local Gemma 4 and exporting one consolidated JSON description file.

## Nodes

- `Arata Analyze Shots (Gemma 4)`
- `Arata Export Gemma Shot JSON`

## Workflow Shape

1. Put video shots in ComfyUI input, or provide an absolute path to a single video file or a folder of video files.
2. Feed `source_path` into `Arata Analyze Shots (Gemma 4)`.
3. Connect `shot_descriptions` into `Arata Export Gemma Shot JSON`.
4. Run the workflow and use the export node download button to fetch the generated `.json` file.

Folder processing is flat and sorted by filename. Nested folders are not traversed.

## Gemma 4 Backend

The node uses local Hugging Face Transformers inference. The default model id is `google/gemma-4-E4B-it`, and the default local model path is `gemma/google-gemma-4-E4B-it` under `ComfyUI/models`. Absolute `model_path` values are used as-is; relative values are resolved from `ComfyUI/models`. Outside ComfyUI, relative paths resolve from this package's local `models` folder.

The backend loads with `local_files_only=True`, so it will not silently download the model during analysis. Download the files before running the node:

```bash
export COMFYUI_DIR="/path/to/ComfyUI"
export MODEL_DIR="$COMFYUI_DIR/models/gemma/google-gemma-4-E4B-it"
# If Hugging Face requires authorization:
# export HF_TOKEN="hf_..."

mkdir -p "$MODEL_DIR"

for f in \
  chat_template.jinja \
  config.json \
  generation_config.json \
  model.safetensors \
  processor_config.json \
  tokenizer.json \
  tokenizer_config.json
do
  if [ -n "${HF_TOKEN:-}" ]; then
    wget -c --show-progress \
      --header="Authorization: Bearer ${HF_TOKEN}" \
      -O "$MODEL_DIR/$f" \
      "https://huggingface.co/google/gemma-4-E4B-it/resolve/main/$f?download=true"
  else
    wget -c --show-progress \
      -O "$MODEL_DIR/$f" \
      "https://huggingface.co/google/gemma-4-E4B-it/resolve/main/$f?download=true"
  fi
done
```

The backend prefers the Gemma 4 multimodal model class and falls back only to compatible image-text classes. Large Gemma 4 variants may require substantial VRAM; use a smaller model or quantized environment if needed.

Video is processed as an ordered sequence of sampled frames:

- Up to 1 frame per second for shots around 60 seconds or shorter.
- A hard cap of 60 sampled frames.
- Longer clips are sampled evenly across the full duration and marked with a warning.

The default visual token budget is `140`, which favors faster video-style analysis. Available budgets are `70`, `140`, `280`, `560`, and `1120`.

The prompt requests the `save_shot_description` tool schema when the installed Transformers/Gemma 4 stack supports tool calls, then still validates the returned fields and falls back to strict JSON parsing and one repair pass.

## Output Format

The export node writes a UTF-8 JSON file under `output/arata_gemma_shots` by default:

```json
{
  "version": 1,
  "generator": "arata_gemma_shots",
  "source_path": "/path/to/shots",
  "model": {
    "id": "google/gemma-4-E4B-it",
    "visual_token_budget": 140
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

The node caches per-video results under the ComfyUI output cache. Re-running the same folder reuses completed cached records when the source file signature, model id, local model path signature, visual token budget, and prompt version match. Failed video records are reported in the current run but are not cached. Local model setup, dependency, device, and inference failures stop the workflow instead of being reported as damaged video files.

## Manual Smoke Test

After installing requirements and model access, run ComfyUI, select a short shot, run the two-node workflow, and confirm that the export JSON contains Russian values under the expected fields.
