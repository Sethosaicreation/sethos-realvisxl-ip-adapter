#!/usr/bin/env bash
set -euo pipefail

: "${DATASET_DIR:?DATASET_DIR must point to the extracted dataset directory}"
: "${OUTPUT_DIR:?OUTPUT_DIR must be the persistent artifact directory}"
: "${INSTANCE_TOKEN:?INSTANCE_TOKEN must match manifest.json, for example skszoeaoki}"

SD_SCRIPTS_DIR="${SD_SCRIPTS_DIR:-/opt/sd-scripts}"
BASE_MODEL="${BASE_MODEL:-SG161222/RealVisXL_V5.0}"
STEPS="${STEPS:-800}"
SEED="${SEED:-20260901}"

test -f "${DATASET_DIR}/manifest.json"
test -f "${SD_SCRIPTS_DIR}/sdxl_train_network.py"
mkdir -p "${OUTPUT_DIR}"

DATASET_CONFIG="${OUTPUT_DIR}/dataset-v2.toml"
python - "${DATASET_DIR}/images" "${DATASET_CONFIG}" <<'PY'
import json
import pathlib
import sys

images = pathlib.Path(sys.argv[1]).resolve()
config = pathlib.Path(sys.argv[2])
manifest = json.loads((images.parent / "manifest.json").read_text(encoding="utf-8"))
if manifest.get("schema") != "sethos.character-lora.dataset.v2":
    raise SystemExit("The dataset was not exported with the ratio-safe v2 schema.")
if len(manifest.get("images", [])) < 16:
    raise SystemExit("The curated dataset contains fewer than 16 images.")
escaped = str(images).replace("\\", "\\\\").replace('"', '\\"')
config.write_text(
    "[general]\n"
    "caption_extension = \".txt\"\n"
    "shuffle_caption = false\n"
    "flip_aug = false\n"
    "color_aug = false\n"
    "random_crop = false\n\n"
    "[[datasets]]\n"
    "resolution = [1024, 1024]\n"
    "batch_size = 1\n"
    "enable_bucket = true\n"
    "bucket_no_upscale = true\n"
    "bucket_reso_steps = 64\n"
    "min_bucket_reso = 512\n"
    "max_bucket_reso = 1536\n\n"
    "  [[datasets.subsets]]\n"
    f"  image_dir = \"{escaped}\"\n"
    "  num_repeats = 1\n",
    encoding="utf-8",
)
PY

cd "${SD_SCRIPTS_DIR}"
accelerate launch --num_cpu_threads_per_process=2 \
  sdxl_train_network.py \
  --pretrained_model_name_or_path="${BASE_MODEL}" \
  --dataset_config="${DATASET_CONFIG}" \
  --output_dir="${OUTPUT_DIR}" \
  --output_name=pytorch_lora_weights \
  --save_model_as=safetensors \
  --network_module=networks.lora \
  --network_dim=32 \
  --network_alpha=16 \
  --network_train_unet_only \
  --gradient_accumulation_steps=2 \
  --gradient_checkpointing \
  --cache_latents \
  --cache_latents_to_disk \
  --cache_text_encoder_outputs \
  --cache_text_encoder_outputs_to_disk \
  --mixed_precision=fp16 \
  --learning_rate=7e-5 \
  --optimizer_type=AdamW8bit \
  --lr_scheduler=constant \
  --lr_warmup_steps=0 \
  --max_train_steps="${STEPS}" \
  --save_every_n_steps=400 \
  --save_last_n_steps=400 \
  --min_snr_gamma=5 \
  --max_data_loader_n_workers=2 \
  --persistent_data_loader_workers \
  --seed="${SEED}" \
  --sdpa

sha256sum "${OUTPUT_DIR}/pytorch_lora_weights.safetensors"
