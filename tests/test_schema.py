from __future__ import annotations

import unittest

from schema import CONTRACT_VERSION, InputError, parse_request


def image_url(slot: str = "source") -> str:
    return (
        "https://sethosaicreation.fr/admin/api/photo-editor-runpod.php"
        f"?action=input&id=pe_{'a' * 24}&slot={slot}&token={'b' * 64}"
    )


def valid_input() -> dict:
    return {
        "input": {
            "contract_version": CONTRACT_VERSION,
            "source_image": image_url(),
            "prompt": "Create a full-body adult portrait.",
            "prompt_template": "full_body",
            "negative_prompt": "watermark",
            "edit_mode": "free",
            "fidelity": "identity",
            "prompt_adherence": "strict",
            "aspect_ratio": "3:4",
            "quality": "standard",
            "content_rating": "standard",
            "adult_content_confirmed": False,
            "rights_confirmed": True,
            "seed": -1,
        }
    }


class SchemaTests(unittest.TestCase):
    def test_valid_request(self) -> None:
        request = parse_request(valid_input())
        self.assertEqual(request.steps, 30)
        self.assertEqual(request.prompt_template, "full_body")
        self.assertEqual(request.prompt_adherence, "strict")

    def test_accepts_influencer_studio_source(self) -> None:
        event = valid_input()
        event["input"]["source_image"] = (
            "https://sethosaicreation.fr/admin/api/influencer-studio.php"
            f"?action=input&id=inf_{'c' * 24}&token={'d' * 64}"
        )
        self.assertIn("influencer-studio.php", parse_request(event).source_image_url)

    def test_accepts_influencer_url_as_style_reference(self) -> None:
        event = valid_input()
        event["input"]["style_image"] = (
            "https://sethosaicreation.fr/admin/api/influencer-studio.php"
            f"?action=input&id=inf_{'c' * 24}&token={'d' * 64}"
        )
        self.assertIn("influencer-studio.php", parse_request(event).style_image_url)

    def test_accepts_signed_home_reference_as_style(self) -> None:
        event = valid_input()
        event["input"]["style_image"] = (
            "https://sethosaicreation.fr/admin/api/influencer-studio.php"
            f"?action=home-input&id=inf_{'c' * 24}&token={'d' * 64}"
        )
        self.assertIn("action=home-input", parse_request(event).style_image_url)

    def test_accepts_character_lora_canary(self) -> None:
        event = valid_input()
        event["input"].update({
            "character_lora": "zoe-aoki-v1",
            "character_lora_sha256": "e" * 64,
            "character_trigger": "skszoeaoki",
            "lora_scale": 0.76,
        })
        request = parse_request(event)
        self.assertEqual(request.character_lora, "zoe-aoki-v1")
        self.assertEqual(request.character_trigger, "skszoeaoki")
        self.assertAlmostEqual(request.lora_scale, 0.76)

    def test_rejects_lora_without_sha256(self) -> None:
        event = valid_input()
        event["input"].update({
            "character_lora": "zoe-aoki-v1",
            "character_trigger": "skszoeaoki",
        })
        with self.assertRaises(InputError):
            parse_request(event)

    def test_rejects_unknown_prompt_adherence(self) -> None:
        event = valid_input()
        event["input"]["prompt_adherence"] = "absolute"
        with self.assertRaises(InputError):
            parse_request(event)

    def test_adult_requires_confirmation(self) -> None:
        event = valid_input()
        event["input"].update({"content_rating": "adult", "prompt_template": "adult_nude_full_body"})
        with self.assertRaises(InputError):
            parse_request(event)

    def test_requires_rights_confirmation(self) -> None:
        event = valid_input()
        event["input"]["rights_confirmed"] = False
        with self.assertRaises(InputError):
            parse_request(event)

    def test_adult_rejects_minor_terms(self) -> None:
        event = valid_input()
        event["input"].update({
            "content_rating": "adult",
            "adult_content_confirmed": True,
            "prompt_template": "adult_nude_full_body",
            "prompt": "A nude 17 year old person",
        })
        with self.assertRaises(InputError):
            parse_request(event)

    def test_style_reference_must_use_signed_slot(self) -> None:
        event = valid_input()
        event["input"]["style_image"] = image_url("style")
        self.assertTrue(parse_request(event).style_image_url)

    def test_rejects_wrong_contract(self) -> None:
        event = valid_input()
        event["input"]["contract_version"] = "wrong"
        with self.assertRaises(InputError):
            parse_request(event)


if __name__ == "__main__":
    unittest.main()
