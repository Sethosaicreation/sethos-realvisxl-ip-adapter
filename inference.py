"""RealVisXL text-to-image generation guided by identity and image references."""

from __future__ import annotations

import math
import hashlib
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
BAKED_REALVIS_CONFIG_DIR = Path(os.getenv("REALVIS_CONFIG_DIR", "/opt/models/realvis-config"))
CHARACTER_LORA_ROOT = Path(os.getenv("CHARACTER_LORA_ROOT", "/runpod-volume/sethos-lora"))

FACE_WEIGHT = "ip-adapter-plus-face_sdxl_vit-h.safetensors"
REFERENCE_WEIGHT = "ip-adapter-plus_sdxl_vit-h.safetensors"
BASE_NEGATIVE = (
    "lowres, low quality, worst quality, blurry, jpeg artifacts, text, watermark, logo, "
    "deformed anatomy, disfigured, extra limbs, missing limbs, fused fingers, extra fingers, "
    "bad hands, malformed feet, duplicate person, plastic skin, waxy skin, beauty filter, "
    "deformed eyes, deformed mouth, distorted face, altered facial structure, different person"
)
FULL_NUDITY_TERMS = re.compile(
    r"\b(?:fully\s+nude|full\s+nudity|completely\s+naked|completly\s+naked|nude|nudity|naked|"
    r"unclothed|entirely\s+unclothed|nu(?:e|es|s)?|nudité)\b",
    re.IGNORECASE,
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


def resolve_pipeline_source(model_dir: Path, config_dir: Path = BAKED_REALVIS_CONFIG_DIR) -> tuple[str, Path, Path | None]:
    """Resolve either a complete Diffusers tree or RealVisXL's FP16 checkpoint.

    RunPod cached models normally expose the full Hugging Face snapshot, but a
    cached host can occasionally expose the large files before the tiny root
    ``model_index.json``. RealVisXL publishes both layouts at the pinned
    revision, so the worker can safely fall back to the FP16 single-file
    checkpoint with configuration baked into the container.
    """
    if (model_dir / "model_index.json").is_file():
        return "diffusers", model_dir, None

    configured = os.getenv("REALVIS_SINGLE_FILE", "").strip()
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured))
    candidates.extend([
        model_dir / "RealVisXL_V5.0_fp16.safetensors",
        model_dir / "realvisxlV50_v50Bakedvae.safetensors",
    ])
    candidates.extend(sorted(model_dir.glob("*fp16*.safetensors")))
    checkpoint = next((candidate for candidate in candidates if candidate.is_file()), None)
    if checkpoint is None:
        raise InferenceError(
            "Cache RealVisXL incomplet : model_index.json et checkpoint FP16 sont absents. "
            "Retirez puis rattachez le modèle mis en cache à l’endpoint RunPod."
        )
    if not (config_dir / "model_index.json").is_file():
        raise InferenceError("Configuration locale RealVisXL absente de l’image du worker.")
    return "single_file", checkpoint, config_dir


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


def _compact_instruction(prompt: str, maximum: int = 950) -> str:
    text = re.sub(r"\s+", " ", prompt).strip()
    text = re.sub(r"^(?:Edit|Use|Extend) Picture 1\.\s*", "", text, flags=re.IGNORECASE)
    if len(text) <= maximum:
        return text
    return text[:maximum].rsplit(" ", 1)[0].rstrip(" ,.;:")


def requests_full_nudity(request: PhotoReferenceRequest) -> bool:
    return request.prompt_template.startswith("adult_nude_") or FULL_NUDITY_TERMS.search(request.prompt) is not None


def prompt_pair(request: PhotoReferenceRequest) -> tuple[str, str]:
    template = TEMPLATE_PROMPTS.get(request.prompt_template, "")
    nude = requests_full_nudity(request)
    if nude:
        mode = (
            "apply the requested complete nudity and do not preserve any garment, underwear or clothing "
            "from the identity reference"
        )
    else:
        mode = {
            "outfit": "apply the requested clothing and styling instead of copying the reference outfit",
            "background": "apply the requested environment instead of copying the reference background",
            "hair": "apply the requested hairstyle and makeup exactly",
            "relight": "apply the requested photographic lighting and atmosphere exactly",
            "free": "apply the requested scene, pose and appearance exactly",
        }[request.edit_mode]
    fidelity = {
        "identity": "preserve the exact facial geometry and recognizable traits from the first reference",
        "balanced": "preserve a clearly recognizable facial identity from the first reference",
        "creative": "preserve recognizable core facial traits while allowing natural variation",
    }[request.fidelity]
    rating = "one clearly adult subject aged 18 or older" if request.content_rating == "adult" else "one adult subject"
    user = _compact_instruction(request.prompt)
    character = f"photo of {request.character_trigger} woman. " if request.character_trigger else ""
    template_context = f" Supporting scene specification: {template}." if template else ""
    primary = (
        f"{character}User instruction, highest priority: {user}.{template_context} RAW personal recent-phone photograph of {rating}. "
        f"{mode}. {fidelity}. The requested pose, body language, clothing state, framing, action and background "
        "must be visibly present; never replace them with a generic neutral standing portrait. Realistic skin texture, "
        "anatomically correct body, natural proportions, sharp detailed face, physically plausible hands and feet, "
        "raw personal-camera texture and coherent practical lighting."
    )
    secondary = (
        f"{character}Follow this instruction literally: {user}.{template_context} {rating}. {mode}. {fidelity}. "
        "Picture 1 controls facial identity only; it must not force the original clothes, pose, crop or background. "
        "Prioritize the requested action and composition while keeping one coherent photorealistic person."
    )
    return primary, secondary


def directed_prompt(request: PhotoReferenceRequest) -> str:
    return prompt_pair(request)[0]


def negative_prompt(request: PhotoReferenceRequest) -> str:
    parts = [request.negative_prompt, BASE_NEGATIVE]
    if requests_full_nudity(request):
        parts.append("clothed, clothes, hoodie, shirt, trousers, underwear, lingerie, bra, panties, swimsuit")
    return ", ".join(part for part in parts if part)


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
            size = min(image.width, image.height, int(max(width, height) * 1.60))
            left = max(0, min(image.width - size, int(center_x - size / 2)))
            top = max(0, min(image.height - size, int(center_y - size / 2)))
            return image.crop((left, top, left + size, top + size))
    except Exception:
        pass
    return _fallback_face_crop(image)


def adapter_strengths(
    request: PhotoReferenceRequest,
    has_style_reference: bool,
    has_character_lora: bool = False,
) -> tuple[float, float, float]:
    face_scale, reference_scale = {
        "identity": (0.92, 0.30 if has_style_reference else 0.12),
        "balanced": (0.82, 0.40 if has_style_reference else 0.18),
        "creative": (0.70, 0.52 if has_style_reference else 0.28),
    }[request.fidelity]
    reference_factor, face_adjustment, guidance_scale = {
        "strict": (0.72, 0.02, 6.5),
        "balanced": (1.0, 0.0, 6.0),
        "reference": (1.18, -0.04, 5.5),
    }[request.prompt_adherence]
    effective_face_scale = min(0.96, max(0.0, face_scale + face_adjustment))
    if has_character_lora:
        # The trained adapter owns identity. Keeping a moderate facial image
        # adapter stabilises freckles and eye colour without pulling the
        # canonical pose back into every generation.
        effective_face_scale = min(0.38, effective_face_scale * 0.48)
    return (
        effective_face_scale,
        min(0.65, max(0.0, reference_scale * reference_factor)),
        guidance_scale,
    )


def resolve_character_lora(request: PhotoReferenceRequest) -> Path | None:
    if not request.character_lora:
        return None
    candidate = CHARACTER_LORA_ROOT / request.character_lora / "pytorch_lora_weights.safetensors"
    try:
        resolved_root = CHARACTER_LORA_ROOT.resolve(strict=True)
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise InferenceError(f"Artefact LoRA privé introuvable : {request.character_lora}.") from error
    if resolved_root not in resolved.parents or not resolved.is_file() or resolved.is_symlink():
        raise InferenceError("Chemin de l’artefact LoRA privé invalide.")
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if not secrets.compare_digest(digest.hexdigest(), request.character_lora_sha256):
        raise InferenceError("Empreinte SHA-256 du LoRA privé incorrecte.")
    return resolved


class RealVisEngine:
    def __init__(self) -> None:
        self._pipeline: Any | None = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._active_lora: tuple[str, float] | None = None

    def _activate_character_lora(
        self, pipeline: Any, request: PhotoReferenceRequest, lora_path: Path | None
    ) -> None:
        requested = (request.character_lora, round(request.lora_scale, 4)) if lora_path is not None else None
        if requested == self._active_lora:
            return
        if self._active_lora is not None:
            pipeline.unload_lora_weights()
            self._active_lora = None
        if lora_path is None:
            return
        pipeline.load_lora_weights(
            str(lora_path.parent),
            weight_name=lora_path.name,
            adapter_name="character",
            local_files_only=True,
        )
        pipeline.set_adapters(["character"], adapter_weights=[request.lora_scale])
        self._active_lora = requested

    def _load(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        with self._load_lock:
            if self._pipeline is not None:
                return self._pipeline
            try:
                import torch
                from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler, StableDiffusionXLPipeline
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
                source_kind, model_source, config_source = resolve_pipeline_source(resolve_model_dir())
                common_load_options = {
                    "image_encoder": image_encoder,
                    "torch_dtype": torch.float16,
                    "local_files_only": True,
                    "low_cpu_mem_usage": True,
                    "use_safetensors": True,
                    "add_watermarker": False,
                }
                if source_kind == "single_file":
                    pipeline = StableDiffusionXLPipeline.from_single_file(
                        str(model_source),
                        config=str(config_source),
                        **common_load_options,
                    )
                else:
                    pipeline = AutoPipelineForText2Image.from_pretrained(
                        str(model_source),
                        **common_load_options,
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
                if hasattr(pipeline, "enable_vae_slicing"):
                    pipeline.enable_vae_slicing()
                elif hasattr(pipeline.vae, "enable_slicing"):
                    pipeline.vae.enable_slicing()
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
            lora_path = resolve_character_lora(request)
            face_scale, reference_scale, guidance_scale = adapter_strengths(
                request, style_path is not None, lora_path is not None
            )
            primary_prompt, secondary_prompt = prompt_pair(request)
            negative = negative_prompt(request)
            with self._inference_lock, torch.inference_mode():
                self._activate_character_lora(pipeline, request, lora_path)
                pipeline.set_ip_adapter_scale([face_scale, reference_scale])
                result = pipeline(
                    prompt=primary_prompt,
                    prompt_2=secondary_prompt,
                    negative_prompt=negative,
                    negative_prompt_2=negative,
                    ip_adapter_image=[identity, secondary],
                    generator=torch.Generator(device="cpu").manual_seed(seed),
                    guidance_scale=guidance_scale,
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
                "guidance_scale": guidance_scale,
                "character_lora": request.character_lora,
                "lora_scale": request.lora_scale if request.character_lora else 0.0,
            }
        except InferenceError:
            raise
        except Exception as error:
            raise InferenceError(f"Génération RealVisXL interrompue : {error}") from error


ENGINE = RealVisEngine()
