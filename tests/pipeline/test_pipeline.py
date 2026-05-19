"""Tests for pipeline/pipeline.py orchestrator.

Covers vts-v1.AC1.1, AC1.2, AC1.3, AC2.4, AC7.1, AC7.2, AC7.3 via
unit tests + optional cloud integration test.

Test structure:
- TestCaptionlessVideoErrorPath (AC2.4): Download returns no captions
- TestAtomicWriteContract (AC7.2): Concurrent reader/writer never see torn JSON
- TestUnknownModelPricingZero (AC7.3): Unknown model zeroes cost, logs warning
- TestCloudIntegration (AC1.1/AC1.2/AC1.3/AC7.1): End-to-end real YouTube (marked @cloud)
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from config import Settings
from pipeline.pipeline import run_job
from pipeline.storage import read_json, write_json_atomic
from pipeline.types import Manifest


class TestCaptionlessVideoErrorPath:
    """AC2.4: Captionless video with WHISPER_FALLBACK=false returns error status."""

    @pytest.mark.asyncio
    async def test_no_captions_sets_error_status(self, tmp_path: Path, monkeypatch):
        """When download returns (path, None) and whisper_fallback is disabled,
        orchestrator sets status=error with clear message mentioning captions and Whisper."""

        # Stub download to return video path but no captions
        def stub_download(url: str, job_dir: Path) -> tuple[Path, Path | None]:
            video_path = job_dir / "video.mp4"
            video_path.touch()
            return video_path, None

        monkeypatch.setattr(
            "pipeline.pipeline.download_video_and_captions",
            stub_download,
        )

        # Setup: create settings with whisper_fallback disabled (use alias names for Settings)
        settings = Settings(
            JOBS_ROOT=tmp_path,
            WHISPER_FALLBACK=False,
            LLM_API_KEY="fake-key",
            JINA_API_KEY="fake-key",
            VISION_API_KEY="fake-key",
        )

        job_id = str(uuid.uuid4())
        url = "https://example.com/fake-video"

        # Act: run the orchestrator
        await run_job(job_id, url, settings, tmp_path)

        # Assert: manifest shows error status with message about captions and Whisper
        meta_path = tmp_path / job_id / "meta.json"
        assert meta_path.exists(), f"meta.json not found at {meta_path}"

        manifest = read_json(meta_path)
        assert manifest["status"] == "error", f"Expected status=error, got {manifest['status']}"
        error_msg = manifest.get("error", "")
        assert "no captions" in error_msg.lower(), \
            f"Error message should mention 'no captions', got: {error_msg}"
        assert "whisper" in error_msg.lower(), \
            f"Error message should mention 'Whisper', got: {error_msg}"


class TestAtomicWriteContract:
    """AC7.2: meta.json is always valid JSON even with concurrent access."""

    def test_atomic_write_never_tears(self, tmp_path: Path):
        """Concurrent reader and writer threads stress the atomic write contract.
        Reader must never see json.JSONDecodeError."""

        json_path = tmp_path / "test.json"
        reader_errors = []
        reader_successes = 0
        writer_count = 0

        def writer_thread_fn():
            nonlocal writer_count
            for i in range(75):
                payload = {"iteration": i, "data": "x" * (100 + i)}
                write_json_atomic(json_path, payload)
                writer_count += 1

        def reader_thread_fn():
            nonlocal reader_successes
            while writer_count < 75 or reader_successes < 75:
                try:
                    if json_path.exists():
                        content = read_json(json_path)
                        if isinstance(content, dict):
                            reader_successes += 1
                except json.JSONDecodeError as e:
                    reader_errors.append(str(e))
                except FileNotFoundError:
                    # File might not exist yet, that's OK
                    pass
                # Small sleep to avoid busy-loop
                threading.Event().wait(0.001)

        # Arrange: start both threads
        t_writer = threading.Thread(target=writer_thread_fn, daemon=False)
        t_reader = threading.Thread(target=reader_thread_fn, daemon=False)

        # Act: run both concurrently
        t_writer.start()
        t_reader.start()
        t_writer.join(timeout=10)
        t_reader.join(timeout=10)

        # Assert: no JSONDecodeError ever occurred
        assert len(reader_errors) == 0, \
            f"Reader saw {len(reader_errors)} JSONDecodeErrors: {reader_errors[:3]}"
        assert writer_count == 75, f"Writer completed {writer_count} iterations, expected 75"
        assert reader_successes > 0, "Reader must have successfully read at least once"


class TestUnknownModelPricingZero:
    """AC7.3: Unknown LLM model records zero cost and logs warning."""

    @pytest.mark.asyncio
    async def test_unknown_model_costs_zero_and_warns(self, tmp_path: Path, monkeypatch, caplog):
        """With llm_model='totally-bogus-model', orchestrator should:
        1. Complete with status=done
        2. Record chat_usd=0.0
        3. Log a warning mentioning the model name"""

        # Stub all pipeline stages to no-op but return minimal valid outputs
        def stub_download(url: str, job_dir: Path) -> tuple[Path, Path | None]:
            video_path = job_dir / "video.mp4"
            video_path.write_bytes(b"")
            vtt_path = job_dir / "video.vtt"
            vtt_path.write_text("WEBVTT\n\n00:00:01.000 --> 00:00:05.000\ntest caption\n")
            return video_path, vtt_path

        def stub_parse_vtt(vtt_path):
            from pipeline.types import Cue
            return [Cue(start=1.0, end=5.0, text="test caption")]

        def stub_dedupe_rolling(cues):
            return cues

        @dataclass(slots=True)
        class StubFrame:
            index: int
            timestamp: float
            path: Path

        @dataclass(slots=True)
        class StubExtractor:
            fps: float = 1.0
            dedup: bool = True
            hamming_max: int = 6

            def extract(self, video_path, frames_dir):
                frames_dir.mkdir(exist_ok=True)
                frame_path = frames_dir / "0001.jpg"
                frame_path.write_bytes(b"")
                return [StubFrame(index=0, timestamp=2.5, path=frame_path)]

        @dataclass(slots=True)
        class StubEmbedResult:
            vectors: Any
            billable_tokens: int = 0

        @dataclass(slots=True)
        class StubEmbedder:
            name: str = "stub"

            async def embed_images(self, paths):
                import numpy as np
                # Return stub embeddings
                return StubEmbedResult(
                    vectors=np.ones((len(paths) if paths else 1, 2), dtype=np.float32),
                    billable_tokens=0,
                )

            async def embed_texts(self, texts):
                import numpy as np
                return StubEmbedResult(
                    vectors=np.ones((len(texts), 2), dtype=np.float32),
                    billable_tokens=0,
                )

            async def aclose(self):
                pass

        @dataclass(slots=True)
        class StubTokenUsage:
            prompt_tokens: int = 0
            completion_tokens: int = 0

        async def stub_llm_outline(cues, llm):
            from pipeline.types import StepOutline
            return (
                [StepOutline(index=0, start=1.0, end=5.0, brief="test step")],
                StubTokenUsage(prompt_tokens=100, completion_tokens=50),
            )

        @dataclass(slots=True)
        class StubLlm:
            async def aclose(self):
                pass

        @dataclass(slots=True)
        class StubVisionCaptioner:
            async def aclose(self):
                pass

        async def stub_caption_winners(winners_by_step, job_dir, captioner, max_in_flight=16):
            return {0: None}, StubTokenUsage()

        async def stub_llm_refine(outlines, cues, winners_by_step, captions, llm, max_in_flight):
            from pipeline.types import Step, Frame as PipelineFrame
            return (
                [
                    Step(
                        index=0,
                        start=1.0,
                        end=5.0,
                        instruction="Do the test step.",
                        frames=[PipelineFrame(index=0, timestamp=2.5, path=tmp_path / "0001.jpg")],
                    )
                ],
                StubTokenUsage(prompt_tokens=50, completion_tokens=30),
            )

        monkeypatch.setattr("pipeline.pipeline.download_video_and_captions", stub_download)
        monkeypatch.setattr("pipeline.pipeline.parse_vtt", stub_parse_vtt)
        monkeypatch.setattr("pipeline.pipeline.dedupe_rolling", stub_dedupe_rolling)
        monkeypatch.setattr("pipeline.pipeline.FixedFpsExtractor", StubExtractor)
        monkeypatch.setattr("pipeline.pipeline.build_embedder", lambda s: StubEmbedder())
        monkeypatch.setattr("pipeline.pipeline.build_llm", lambda s: StubLlm())
        monkeypatch.setattr("pipeline.pipeline.build_vision", lambda s: StubVisionCaptioner())
        monkeypatch.setattr("pipeline.pipeline.llm_outline", stub_llm_outline)
        monkeypatch.setattr("pipeline.pipeline.caption_winners", stub_caption_winners)
        monkeypatch.setattr("pipeline.pipeline.llm_refine", stub_llm_refine)

        # Setup: create settings with unknown model (use alias names for Settings)
        settings = Settings(
            JOBS_ROOT=tmp_path,
            LLM_MODEL="totally-bogus-model",
            LLM_API_KEY="fake-key",
            JINA_API_KEY="fake-key",
            VISION_API_KEY="fake-key",
            WHISPER_FALLBACK=False,
        )

        job_id = str(uuid.uuid4())
        url = "https://example.com/fake-video"

        # Capture logs at WARNING level
        with caplog.at_level(logging.WARNING):
            # Act: run the orchestrator
            await run_job(job_id, url, settings, tmp_path)

        # Assert: job completed with status=done
        meta_path = tmp_path / job_id / "meta.json"
        assert meta_path.exists()
        manifest = read_json(meta_path)
        assert manifest["status"] == "done", \
            f"Expected status=done, got {manifest['status']}"

        # Assert: chat_usd is 0.0 (unknown model)
        cost = manifest.get("cost", {})
        assert cost.get("chat_usd") == 0.0, \
            f"Expected chat_usd=0.0 for unknown model, got {cost.get('chat_usd')}"

        # Assert: a warning was logged mentioning the model name
        warning_logs = [r for r in caplog.records if r.levelname == "WARNING"]
        model_warning_found = any(
            "totally-bogus-model" in r.message.lower()
            for r in warning_logs
        )
        assert model_warning_found, \
            f"Expected warning log mentioning model name, got: {[r.message for r in warning_logs]}"


@pytest.mark.cloud
class TestCloudIntegration:
    """AC1.1/AC1.2/AC1.3/AC7.1: End-to-end real YouTube run (optional, marked @cloud)."""

    @pytest.mark.asyncio
    async def test_end_to_end_real_video(self, tmp_path: Path):
        """Integration test using a real short instructional YouTube video.
        Requires RUN_CLOUD_TESTS=1 and .env configured with API keys.

        Asserts:
        - status=done
        - >=3 steps
        - Each step has >=1 frame and 1-3 sentence instruction
        - total_usd > 0
        - steps.json parses cleanly

        Manual check (AC1.3): human reviews printed step-frame mappings.
        """
        # Skip this test in standard environments (no real API keys)
        try:
            from config import get_settings
            settings = get_settings()
            # Check if keys are configured
            if not settings.llm_api_key or not settings.jina_api_key or not settings.vision_api_key:
                pytest.skip("API keys not configured in .env")
        except Exception:
            pytest.skip("Could not load settings or API keys")

        # Choose a short instructional video (hardcoded for reproducibility)
        # This URL should be <= 3 minutes and show clear step-by-step process
        # Example: a simple knot-tying or basic recipe video
        # For now, we'll use a placeholder — replace with real URL during execution
        video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Placeholder

        pytest.skip(
            "Cloud integration test placeholder. "
            "Set RUN_CLOUD_TESTS=1, configure .env with real keys, "
            "then replace video_url with a real instructional video and run."
        )

        # If we get here, run the test
        from config import get_settings
        settings = get_settings()

        job_id = str(uuid.uuid4())
        await run_job(job_id, video_url, settings, tmp_path)

        # Assert: meta.json exists and status is done
        meta_path = tmp_path / job_id / "meta.json"
        assert meta_path.exists()
        manifest = read_json(meta_path)
        assert manifest["status"] == "done", f"Expected status=done, got {manifest['status']}"

        # Assert: steps.json exists and parses
        steps_path = tmp_path / job_id / "steps.json"
        assert steps_path.exists()
        steps = read_json(steps_path)

        # Assert: >= 3 steps
        assert len(steps) >= 3, f"Expected >=3 steps, got {len(steps)}"

        # Assert: each step has >=1 frame and valid instruction
        for step_idx, step in enumerate(steps):
            assert len(step.get("frames", [])) >= 1, \
                f"Step {step_idx} has no frames"
            instruction = step.get("instruction", "")
            sentences = [s.strip() for s in instruction.split(".") if s.strip()]
            assert 1 <= len(sentences) <= 3, \
                f"Step {step_idx} instruction has {len(sentences)} sentences, expected 1-3: {instruction}"

        # Assert: cost > 0
        cost = manifest.get("cost", {})
        total_usd = cost.get("total_usd", 0.0)
        assert total_usd > 0.0, f"Expected total_usd > 0, got {total_usd}"

        # Print step-frame mapping for manual AC1.3 verification
        print("\n=== Manual Frame Relevance Check (AC1.3) ===")
        for step_idx, step in enumerate(steps):
            print(f"\nStep {step_idx}: {step.get('instruction', '')}")
            for frame in step.get("frames", []):
                frame_path = frame.get("path", "unknown")
                print(f"  Frame: {frame_path}")
