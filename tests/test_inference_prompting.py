from __future__ import annotations

import unittest

from inference import adapter_strengths, negative_prompt, prompt_pair
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


if __name__ == "__main__":
    unittest.main()
