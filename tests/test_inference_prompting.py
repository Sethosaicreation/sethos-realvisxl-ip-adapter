from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from inference import (
    InferenceError,
    adapter_strengths,
    materialize_diffusers_layout,
    negative_prompt,
    prompt_pair,
    resolve_pipeline_source,
)
from schema import CONTRACT_VERSION, PhotoReferenceRequest


def request(**overrides: object) -> PhotoReferenceRequest:
    values: dict[str, object] = {
        "contract_version": CONTRACT_VERSION,
        "source_image_url": "https://sethosaicreation.fr/source",
        "style_image_url": "",
        "prompt": "full body, completely naked in the requested dynamic pose",
        "prompt_template": "",
        "negative_prompt": "",
        "edit_mode": "outfit",
        "fidelity": "identity",
        "prompt_adherence": "strict",
        "aspect_ratio": "3:4",
        "quality": "standard",
        "content_rating": "adult",
        "seed": -1,
    }
    values.update(overrides)
    return PhotoReferenceRequest(**values)  # type: ignore[arg-type]


class PromptingTests(unittest.TestCase):
    def test_complete_diffusers_snapshot_is_preferred(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model"
            config = root / "config"
            model.mkdir()
            config.mkdir()
            (model / "model_index.json").write_text("{}", encoding="utf-8")
            kind, source, fallback = resolve_pipeline_source(model, config)
            self.assertEqual(kind, "diffusers")
            self.assertEqual(source, model)
            self.assertIsNone(fallback)

    def test_missing_model_index_uses_fp16_single_file(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model"
            config = root / "config"
            model.mkdir()
            config.mkdir()
            checkpoint = model / "RealVisXL_V5.0_fp16.safetensors"
            checkpoint.write_bytes(b"test")
            (config / "model_index.json").write_text("{}", encoding="utf-8")
            kind, source, fallback = resolve_pipeline_source(model, config)
            self.assertEqual(kind, "single_file")
            self.assertEqual(source, checkpoint)
            self.assertEqual(fallback, config)

    def test_weight_only_diffusers_cache_is_merged_with_baked_config(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model"
            config = root / "config"
            merged = root / "merged"
            model.mkdir()
            config.mkdir()
            (config / "model_index.json").write_text("{}", encoding="utf-8")
            for component in ("unet", "vae", "text_encoder", "text_encoder_2"):
                (model / component).mkdir()
                (model / component / "weights.safetensors").write_bytes(component.encode())
                (config / component).mkdir()
                (config / component / "config.json").write_text("{}", encoding="utf-8")
            result = materialize_diffusers_layout(model, config, merged)
            self.assertEqual(result, merged)
            self.assertTrue((merged / "model_index.json").is_file())
            self.assertTrue((merged / "unet" / "weights.safetensors").is_symlink())

    def test_incomplete_cache_fails_with_actionable_error(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model"
            config = root / "config"
            model.mkdir()
            config.mkdir()
            (config / "model_index.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(InferenceError, "Cache RealVisXL incomplet"):
                resolve_pipeline_source(model, config)

    def test_user_instruction_leads_both_sdxl_prompts(self) -> None:
        primary, secondary = prompt_pair(request())
        self.assertTrue(primary.startswith("User instruction, highest priority: full body"))
        self.assertTrue(secondary.startswith("Follow this instruction literally: full body"))
        self.assertIn("never replace them with a generic neutral standing portrait", primary)

    def test_full_nudity_does_not_preserve_reference_clothing(self) -> None:
        primary, secondary = prompt_pair(request())
        self.assertIn("do not preserve any garment", primary)
        self.assertIn("must not force the original clothes", secondary)
        self.assertIn("hoodie", negative_prompt(request()))

    def test_strict_identity_prioritizes_face_and_text(self) -> None:
        face, reference, guidance = adapter_strengths(request(), False)
        self.assertGreaterEqual(face, 0.9)
        self.assertLessEqual(reference, 0.1)
        self.assertGreaterEqual(guidance, 6.5)

    def test_reference_mode_increases_second_reference_influence(self) -> None:
        strict = adapter_strengths(request(prompt_adherence="strict"), True)
        reference = adapter_strengths(request(prompt_adherence="reference"), True)
        self.assertGreater(reference[1], strict[1])
        self.assertLess(reference[2], strict[2])

    def test_lora_trigger_leads_both_prompts(self) -> None:
        primary, secondary = prompt_pair(request(
            character_lora="zoe-aoki-v1",
            character_lora_sha256="a" * 64,
            character_trigger="skszoeaoki",
            lora_scale=0.78,
        ))
        self.assertTrue(primary.startswith("photo of skszoeaoki woman."))
        self.assertTrue(secondary.startswith("photo of skszoeaoki woman."))

    def test_lora_keeps_face_authoritative_without_letting_style_dominate(self) -> None:
        normal = adapter_strengths(request(), True)
        canary = adapter_strengths(request(character_lora="zoe-aoki-v1"), True, True)
        self.assertLess(canary[0], normal[0])
        self.assertGreaterEqual(canary[0], 0.54)
        self.assertLessEqual(canary[0], 0.64)
        self.assertLessEqual(canary[1], 0.30)


if __name__ == "__main__":
    unittest.main()
