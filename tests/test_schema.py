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
            "aspect_ratio": "3:4",
            "quality": "standard",
            "content_rating": "standard",
            "adult_content_confirmed": False,
            "seed": -1,
        }
    }


class SchemaTests(unittest.TestCase):
    def test_valid_request(self) -> None:
        request = parse_request(valid_input())
        self.assertEqual(request.steps, 30)
        self.assertEqual(request.prompt_template, "full_body")

    def test_adult_requires_confirmation(self) -> None:
        event = valid_input()
        event["input"].update({"content_rating": "adult", "prompt_template": "adult_nude_full_body"})
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
