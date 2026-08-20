"""RealVisXL text-to-image generation guided by identity and image references."""

from __future__ import annotations

import math
import os
import re
import secrets
import threading
from pathlib import Path
from typing import Any

from PIL import Image

from schema import PhotoReferenceRequest


MODEL_ID = os.getenv("REALVIS_MODEL_ID", "SG161222/RealVisXL_V5.0")
MODEL_REVISION = os.getenv("REALVIS_MODEL_REVISION", "ac93e0dda1f6d448cae19bbfab8c5e720a5e48bc")
ADAPTER_ID = os.getenv("IP_ADAPTER_MODEL_ID", "h94/IP-Adapter")
ADAPTER_REVISION = os.getenv("IP_ADAPTER_MODEL_REVISION", "018e402774aeeddd60609b4ecdb7e298259dc729")
MODEL_CACHE_ROOT = Path("/runpod-volume/huggingface-cache/hub")
BAKED_ADAPTER_DIR = Path(os.getenv("IP_ADAPTER_MODEL_DIR", "/opt/models/ip-adapter"))

FACE_WEIGHT = "ip-adapter-plus-face_sdxl_vit-h.safetensors"
REFERENCE_WEIGHT = "ip-adapter-plus_sdxl_vit-h.safetensors"
BASE_NEGATIVE = (
    "lowres, low quality, worst quality, blurry, jpeg artifacts, text, watermark, logo, "
    "deformed anatomy, disfigured, extra limbs, missing limbs, fused fingers, extra fingers, "
    "bad hands, malformed feet, asymmetrical eyes, duplicate person, plastic skin"
)
TEMPLATE_PROMPTS = {
    "full_body": "full-body photograph of one adult person, complete body from head to toe, both hands and feet visible",
    "identity_body": "full-body photograph using the first reference for facial identity and the second reference for body proportions and pose",
    "outpainting": "complete full-body reconstruction of the adult portrait, naturally extending the visible person from head to toe",
    "adult_nude_full_body": "tasteful non-sexual fully nude full-body photograph of one consenting adult aged 18 or older, natural adult anatomy",
    "adult_nude_identity_body": "tasteful non-sexual fully nude full-body photograph of one consenting adult aged 18 or older, first reference for identity and second reference for body and pose",
    "adult_nude_outpainting": "tasteful non-sexual fully nude full-body reconstruction of one consenting adult aged 18 or older, naturally extended from head to toe",
    "adult_nude_art_studio": "tasteful non-sexual fully nude consenting adult aged 18 or older posing naturally in a bright fine-art photography studio with soft north-window light",
    "adult_nude_morning_room": "tasteful non-sexual fully nude consenting adult aged 18 or older standing naturally in a private sunlit bedroom in the morning",
    "adult_nude_spa": "tasteful non-sexual fully nude consenting adult aged 18 or older in a private contemporary spa after a shower, light steam and water droplets",
    "adult_nude_private_nature": "tasteful non-sexual fully nude consenting adult aged 18 or older in a secluded private natural setting at golden hour",
}


class InferenceError(RuntimeError):
    pass


def _snapshot_dir(model_id: str, revision: str, configured: str = "") -> Path:
    if configured:
        path = Path(configured)
        if path.is_dir():
            return path
    if "/" not in model_id:
        raise InferenceError(f"Identifiant de modèle invalide : {model_id}.")
    organization, name = model_id.split("/", 1)
    root = MODEL_CACHE_ROOT / f"models--{organization}--{name}"
    pinned = root / "snapshots" / revision
    if pinned.is_dir():
        return pinned
    reference = root / "refs" / "main"
    if reference.is_file():
        candidate_revision = reference.read_text(encoding="utf-8").strip()
        candidate = root / "snapshots" / candidate_revision
        if candidate_revision and candidate.is_dir():
            return candidate
    snapshots = sorted((root / "snapshots").glob("*"), key=lambda path: path.stat().st_mtime, reverse=True) \
        if (root / "snapshots").is_dir() else []
    if snapshots:
        return snapshots[0]
    raise InferenceError(f"Poids {model_id} introuvables dans le cache RunPod.")


def resolve_model_dir() -> Path:
    return _snapshot_dir(MODEL_ID, MODEL_REVISION, os.getenv("REALVIS_MODEL_DIR", "").strip())


def resolve_adapter_dir() -> Path:
    if BAKED_ADAPTER_DIR.is_dir():
        return BAKED_ADAPTER_DIR
    return _snapshot_dir(ADAPTER_ID, ADAPTER_REVISION, os.getenv("IP_ADAPTER_OVERRIDE_DIR", "").strip())


def output_dimensions(source: Image.Image, request: PhotoReferenceRequest) -> tuple[int, int]:
    if request.aspect_ratio == "source":
        ratio = source.width / source.height
    else:
        left, right = request.aspect_ratio.split(":", 1)
        ratio = int(left) / int(right)
    pixels = {"preview": 786_432, "standard": 1_048_576, "quality": 1_376_256}[request.quality]
    height = math.sqrt(pixels / ratio)
    width = height * ratio
    width = max(512, int(round(width / 64)) * 64)
    height = max(512, int(round(height / 64)) * 64)
    while width * height > pixels:
        if width >= height and width > 512:
            width -= 64
        elif height > 512:
            height -= 64
        else:
            break
    return width, height


def _compact_instruction(prompt: str, maximum: int = 420) -> str:
    text = re.sub(r"\s+", " ", prompt).strip()
    text = re.sub(r"^(?:Edit|Use|Extend) Picture 1\.\s*", "", text, flags=re.IGNORECASE)
    if len(text) <= maximum:
        return text
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    priority_words = (
        "requested", "appearance", "background", "studio", "bedroom", "spa", "shower", "nature", "lakeside",
        "pose", "standing", "sitting", "walking", "lighting", "lens", "nude", "unclothed", "full-body", "head to toe",
    )
    selected: list[str] = []
    for sentence in sentences:
        lowered = sentence.lower()
        if any(word in lowered for word in priority_words) and sentence not in selected:
            selected.append(sentence)
    if not selected:
        selected = sentences[:2]
    compact = " ".join(selected)
    return compact[:maximum].rsplit(" ", 1)[0].rstrip(" ,.;:")


def directed_prompt(request: PhotoReferenceRequest) -> str:
    template = TEMPLATE_PROMPTS.get(request.prompt_template, "")
    mode = {
        "outfit": "requested clothing and styling",
        "background": "requested environment and background",
        "hair": "requested hairstyle and makeup",
        "relight": "requested photographic lighting and atmosphere",
        "free": "requested scene and appearance",
    }[request.edit_mode]
    fidelity = {
        "identity": "strong facial identity match to the first reference",
        "balanced": "recognizable facial identity with natural variation",
        "creative": "recognizable identity with greater compositional freedom",
    }[request.fidelity]
    rating = "clearly adult subject aged 18 or older, " if request.content_rating == "adult" else "adult subject, "
    user = _compact_instruction(request.prompt)
    subject = template or user
    extra = f", {user}" if template and user.lower() not in template.lower() else ""
    return (
        f"RAW photorealistic editorial photograph, {rating}{subject}{extra}, {mode}, {fidelity}, "
        "one coherent person, realistic skin texture, anatomically correct body, natural proportions, "
        "sharp detailed face, physically plausible hands and feet, professional photography, natural light"
    )


def negative_prompt(request: PhotoReferenceRequest) -> str:
    return f"{request.negative_prompt}, {BASE_NEGATIVE}" if request.negative_prompt else BASE_NEGATIVE


def _fallback_face_crop(image: Image.Image) -> Image.Image:
    width, height = image.size
    size = min(width, height)
    left = max(0, (width - size) // 2)
    top = max(0, min(height - size, int(height * 0.04) if height > width else (height - size) // 2))
    return image.crop((left, top, left + size, top + size))


def face_reference(image: Image.Image) -> Image.Image:
    try:
        import cv2
        import numpy as np

        rgb = np.asarray(image.convert("RGB"))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        cascade = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"))
        minimum = max(32, min(image.size) // 10)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=4, minSize=(minimum, minimum))
        if len(faces):
            x, y, width, height = max(faces, key=lambda face: int(face[2]) * int(face[3]))
            center_x = x + width / 2
            center_y = y + height / 2
            size = min(image.width, image.height, int(max(width, height) * 2.15))
            left = max(0, min(image.width - size, int(center_x - size / 2)))
            top = max(0, min(image.height - size, int(center_y - size / 2)))
            return image.crop((left, top, left + size, top + size))
    except Exception:
        pass
    return _fallback_face_crop(image)


class RealVisEngine:
    def __init__(self) -> None:
        self._pipeline: Any | None = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    def _load(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        with self._load_lock:
            if self._pipeline is not None:
                return self._pipeline
            try:
                import torch
                from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler
                from transformers import CLIPVisionModelWithProjection

                if not torch.cuda.is_available():
                    raise InferenceError("GPU CUDA indisponible sur le worker RunPod.")
                adapter_dir = resolve_adapter_dir()
                image_encoder_dir = adapter_dir / "models" / "image_encoder"
                if not image_encoder_dir.is_dir():
                    raise InferenceError("Encodeur visuel IP-Adapter introuvable dans l’image du worker.")
                image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                    str(image_encoder_dir), torch_dtype=torch.float16, local_files_only=True,
                )
                pipeline = AutoPipelineForText2Image.from_pretrained(
                    str(resolve_model_dir()),
                    image_encoder=image_encoder,
                    torch_dtype=torch.float16,
                    local_files_only=True,
                    low_cpu_mem_usage=True,
                    use_safetensors=True,
                    add_watermarker=False,
                )
                pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                    pipeline.scheduler.config, algorithm_type="sde-dpmsolver++", use_karras_sigmas=True,
                )
                pipeline.load_ip_adapter(
                    str(adapter_dir),
                    subfolder="sdxl_models",
                    weight_name=[FACE_WEIGHT, REFERENCE_WEIGHT],
                    local_files_only=True,
                )
                pipeline.to("cuda")
                pipeline.enable_vae_slicing()
                pipeline.set_progress_bar_config(disable=True)
                self._pipeline = pipeline
                return pipeline
            except InferenceError:
                raise
            except Exception as error:
                raise InferenceError(f"Chargement de RealVisXL + IP-Adapter impossible : {error}") from error

    def generate(
        self, request: PhotoReferenceRequest, source_path: Path, style_path: Path | None
    ) -> tuple[Image.Image, dict[str, Any]]:
        try:
            import torch

            pipeline = self._load()
            source = Image.open(source_path).convert("RGB")
            secondary = Image.open(style_path).convert("RGB") if style_path is not None else source
            identity = face_reference(source)
            seed = request.seed if request.seed >= 0 else secrets.randbelow(2_147_483_648)
            width, height = output_dimensions(source, request)
            face_scale, reference_scale = {
                "identity": (0.82, 0.38 if style_path is not None else 0.28),
                "balanced": (0.70, 0.46 if style_path is not None else 0.34),
                "creative": (0.56, 0.55 if style_path is not None else 0.40),
            }[request.fidelity]
            with self._inference_lock, torch.inference_mode():
                pipeline.set_ip_adapter_scale([face_scale, reference_scale])
                result = pipeline(
                    prompt=directed_prompt(request),
                    negative_prompt=negative_prompt(request),
                    ip_adapter_image=[identity, secondary],
                    generator=torch.Generator(device="cpu").manual_seed(seed),
                    guidance_scale=5.0,
                    num_inference_steps=request.steps,
                    num_images_per_prompt=1,
                    width=width,
                    height=height,
                )
            output = result.images[0]
            return output, {
                "seed": seed,
                "width": output.width,
                "height": output.height,
                "face_scale": face_scale,
                "reference_scale": reference_scale,
            }
        except InferenceError:
            raise
        except Exception as error:
            raise InferenceError(f"Génération RealVisXL interrompue : {error}") from error


ENGINE = RealVisEngine()
