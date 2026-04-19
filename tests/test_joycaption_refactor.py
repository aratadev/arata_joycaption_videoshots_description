from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PARENT_ROOT = PACKAGE_ROOT.parent
if str(PARENT_ROOT) not in sys.path:
    sys.path.insert(0, str(PARENT_ROOT))

from arata_gemma_videoshots_description.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from arata_gemma_videoshots_description.nodes.analyze_shots import ArataJoyCaptionShotAnalyze
from arata_gemma_videoshots_description.nodes.export_json import ArataJoyCaptionShotJsonExport
from arata_gemma_videoshots_description.services.export_service import JoyCaptionShotJsonExportService
from arata_gemma_videoshots_description.services.joycaption_analysis_service import (
    DEFAULT_JOYCAPTION_MODEL_ID,
    DEFAULT_JOYCAPTION_MODEL_PATH,
    PROMPT_VERSION,
    JoyCaptionShotAnalysisService,
)
from arata_gemma_videoshots_description.services.joycaption_backend import (
    DESCRIPTION_FIELDS,
    JOYCAPTION_REQUIRED_FILES,
    JoyCaptionLocalBackend,
)
from arata_gemma_videoshots_description.utils.models import (
    JoyCaptionShotDescriptionResult,
    SampledKeyframe,
    VideoDescriptionRecord,
    VideoMetadata,
)
from arata_gemma_videoshots_description.utils.video_sampling import VideoFrameSample


VALID_DESCRIPTION = {
    "summary": "summary",
    "actions": ["action"],
    "objects": ["object"],
    "environment": "environment",
    "atmosphere": "atmosphere",
    "shot_scale": "medium shot",
    "camera_motion": "static",
    "temporal_notes": ["0.000s: action"],
}


class JoyCaptionBackendValidationTests(unittest.TestCase):
    def test_local_model_validation_accepts_single_safetensors_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp)
            self._write_required_files(model_path)
            (model_path / "model.safetensors").write_text("", encoding="utf-8")

            backend = JoyCaptionLocalBackend(DEFAULT_JOYCAPTION_MODEL_ID, model_path, "cpu", 512, PROMPT_VERSION)

            backend._validate_local_model_path()

    def test_local_model_validation_accepts_sharded_safetensors_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp)
            self._write_required_files(model_path)
            (model_path / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
            (model_path / "model-00001-of-00004.safetensors").write_text("", encoding="utf-8")

            backend = JoyCaptionLocalBackend(DEFAULT_JOYCAPTION_MODEL_ID, model_path, "cpu", 512, PROMPT_VERSION)

            backend._validate_local_model_path()

    def test_local_model_validation_reports_missing_required_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp)
            (model_path / "config.json").write_text("{}", encoding="utf-8")
            (model_path / "model.safetensors").write_text("", encoding="utf-8")

            backend = JoyCaptionLocalBackend(DEFAULT_JOYCAPTION_MODEL_ID, model_path, "cpu", 512, PROMPT_VERSION)

            with self.assertRaisesRegex(RuntimeError, "Missing files"):
                backend._validate_local_model_path()

    def test_local_model_validation_reports_missing_folder(self) -> None:
        backend = JoyCaptionLocalBackend(
            DEFAULT_JOYCAPTION_MODEL_ID,
            Path("/definitely/missing/joycaption"),
            "cpu",
            512,
            PROMPT_VERSION,
        )

        with self.assertRaisesRegex(RuntimeError, "model folder not found"):
            backend._validate_local_model_path()

    def _write_required_files(self, model_path: Path) -> None:
        for filename in JOYCAPTION_REQUIRED_FILES:
            (model_path / filename).write_text("{}", encoding="utf-8")


class InvalidSynthesisBackend(JoyCaptionLocalBackend):
    def _caption_frame(self, image: Any, keyframe: SampledKeyframe, output_language: str) -> str:
        return f"Caption for frame {keyframe.index}"

    def _synthesize_shot_description(
        self,
        video_path: Path,
        metadata: VideoMetadata,
        captions: tuple[Any, ...],
        output_language: str,
    ) -> str:
        return "not json"


class JoyCaptionBackendFallbackTests(unittest.TestCase):
    def test_invalid_second_pass_falls_back_to_required_description_fields(self) -> None:
        backend = InvalidSynthesisBackend(DEFAULT_JOYCAPTION_MODEL_ID, Path("."), "cpu", 512, PROMPT_VERSION)
        metadata = VideoMetadata(source_video_path="shot.mp4", fps=24.0, total_frames=24, duration_sec=1.0)
        samples = (
            VideoFrameSample(SampledKeyframe(index=1, frame_index=0, timestamp_sec=0.0), object()),
            VideoFrameSample(SampledKeyframe(index=2, frame_index=12, timestamp_sec=0.5), object()),
        )

        description = backend.describe_video(Path("shot.mp4"), metadata, samples, "Russian")

        self.assertIsInstance(description, dict)
        self.assertTrue(all(field in description for field in DESCRIPTION_FIELDS))
        self.assertEqual(description["temporal_notes"][0], "0.000s: Caption for frame 1")


class FakeBackend:
    describe_calls = 0
    repair_calls = 0

    def describe_video(
        self,
        video_path: Path,
        metadata: VideoMetadata,
        frame_samples: tuple[VideoFrameSample, ...],
        output_language: str,
    ) -> dict[str, Any] | str:
        self.describe_calls += 1
        return dict(VALID_DESCRIPTION)

    def repair_description(self, invalid_output: Any, error: Exception, output_language: str) -> dict[str, Any] | str:
        self.repair_calls += 1
        return dict(VALID_DESCRIPTION)


class JoyCaptionServiceTests(unittest.TestCase):
    def test_service_preserves_sampling_and_reuses_completed_cache_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video_path = root / "shot001.mp4"
            video_path.write_bytes(b"fake")
            model_path = (root / "model").resolve()
            model_path.mkdir()
            cache_dir = root / "cache"
            backend = FakeBackend()
            factory_calls = 0

            def backend_factory(model_id: str, path: Path, device: str, caption_max_tokens: int) -> FakeBackend:
                nonlocal factory_calls
                factory_calls += 1
                self.assertEqual(model_id, DEFAULT_JOYCAPTION_MODEL_ID)
                self.assertEqual(path, model_path)
                self.assertEqual(device, "cpu")
                self.assertEqual(caption_max_tokens, 512)
                return backend

            def metadata_probe(path: str) -> VideoMetadata:
                return VideoMetadata(source_video_path=path, fps=24.0, total_frames=24, duration_sec=1.0)

            def frame_sampler(path: str, metadata: VideoMetadata) -> tuple[VideoFrameSample, ...]:
                return (
                    VideoFrameSample(SampledKeyframe(index=1, frame_index=0, timestamp_sec=0.0), object()),
                )

            service = JoyCaptionShotAnalysisService(
                backend_factory=backend_factory,
                metadata_probe=metadata_probe,
                frame_sampler=frame_sampler,
                cache_dir=cache_dir,
            )

            first = service.analyze(
                source_path=str(video_path),
                model_id=DEFAULT_JOYCAPTION_MODEL_ID,
                model_path=str(model_path),
                device="cpu",
                caption_max_tokens=512,
            )
            second = service.analyze(
                source_path=str(video_path),
                model_id=DEFAULT_JOYCAPTION_MODEL_ID,
                model_path=str(model_path),
                device="cpu",
                caption_max_tokens=512,
            )

            self.assertEqual(factory_calls, 1)
            self.assertEqual(backend.describe_calls, 1)
            self.assertEqual(first.caption_max_tokens, 512)
            self.assertEqual(first.prompt_version, PROMPT_VERSION)
            self.assertEqual(first.videos[0].keyframes[0].frame_index, 0)
            self.assertEqual(second.videos[0].description["summary"], "summary")


class JoyCaptionNodeAndExportTests(unittest.TestCase):
    def test_node_mappings_use_new_public_ids_without_gemma_aliases(self) -> None:
        self.assertIn("ArataJoyCaptionShotAnalyze", NODE_CLASS_MAPPINGS)
        self.assertIn("ArataJoyCaptionShotJsonExport", NODE_CLASS_MAPPINGS)
        self.assertNotIn("ArataGemmaShotAnalyze", NODE_CLASS_MAPPINGS)
        self.assertNotIn("ArataGemmaShotJsonExport", NODE_CLASS_MAPPINGS)
        self.assertEqual(
            NODE_DISPLAY_NAME_MAPPINGS["ArataJoyCaptionShotAnalyze"],
            "Arata Analyze Shots (JoyCaption Beta One)",
        )

    def test_node_input_and_return_types_are_joycaption_specific(self) -> None:
        analyze_inputs = ArataJoyCaptionShotAnalyze.INPUT_TYPES()["required"]
        export_inputs = ArataJoyCaptionShotJsonExport.INPUT_TYPES()["required"]

        self.assertEqual(ArataJoyCaptionShotAnalyze.RETURN_TYPES, ("ARATA_JOYCAPTION_SHOT_DESCRIPTIONS",))
        self.assertEqual(analyze_inputs["model_path"][1]["default"], DEFAULT_JOYCAPTION_MODEL_PATH)
        self.assertEqual(analyze_inputs["caption_max_tokens"][1]["default"], "512")
        self.assertEqual(export_inputs["shot_descriptions"][0], "ARATA_JOYCAPTION_SHOT_DESCRIPTIONS")
        self.assertEqual(export_inputs["output_subdirectory"][1]["default"], "arata_joycaption_shots")

    def test_export_defaults_use_joycaption_paths_and_payload_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp)
            result = JoyCaptionShotDescriptionResult(
                source_path="/input/shot001.mp4",
                generated_at="2026-04-19T00:00:00+00:00",
                model_id=DEFAULT_JOYCAPTION_MODEL_ID,
                caption_max_tokens=512,
                prompt_version=PROMPT_VERSION,
                videos=(
                    VideoDescriptionRecord(
                        source_video_path="/input/shot001.mp4",
                        filename="shot001.mp4",
                        status="completed",
                        description=dict(VALID_DESCRIPTION),
                    ),
                ),
            )

            exported = JoyCaptionShotJsonExportService(output_root=output_root).export(
                shot_descriptions=result,
                output_subdirectory="",
                filename_stem="",
                overwrite_existing=False,
            )

            exported_path = Path(exported.descriptions_file.file_path)
            self.assertEqual(exported_path.parent.name, "arata_joycaption_shots")
            self.assertEqual(exported_path.name, "shot001_joycaption_shots.json")
            payload = json.loads(exported_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["generator"], "arata_joycaption_shots")
            self.assertEqual(payload["model"]["caption_max_tokens"], 512)
            self.assertNotIn("visual_token_budget", payload["model"])


if __name__ == "__main__":
    unittest.main()
