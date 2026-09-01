#!/usr/bin/env bash
set -euo pipefail

: "${DATASET_DIR:?DATASET_DIR must point to the extracted dataset directory}"
: "${OUTPUT_DIR:?OUTPUT_DIR must be the persistent artifact directory}"
: "${INSTANCE_TOKEN:?INSTANCE_TOKEN must match manifest.json, for example skszoeaoki}"

DIFFUSERS_SOURCE_DIR="${DIFFUSERS_SOURCE_DIR:-/opt/diffusers-source}"
BASE_MODEL="${BASE_MODEL:-SG161222/RealVisXL_V5.0}"
STEPS="${STEPS:-1600}"
SEED="${SEED:-20260901}"

test -f "${DATASET_DIR}/images/metadata.jsonl"
test -f "${DATASET_DIR}/manifest.json"
test -f "${DIFFUSERS_SOURCE_DIR}/examples/text_to_image/train_text_to_image_lora_sdxl.py"
mkdir -p "${OUTPUT_DIR}"

accelerate launch \
  "${DIFFUSERS_SOURCE_DIR}/examples/text_to_image/train_text_to_image_lora_sdxl.py" \
  --pretrained_model_name_or_path="${BASE_MODEL}" \
  --train_data_dir="${DATASET_DIR}/images" \
  --caption_column="text" \
  --resolution=1024 \
  --center_crop \
  --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --mixed_precision=fp16 \
  --learning_rate=1e-4 \
  --lr_scheduler=constant \
  --lr_warmup_steps=0 \
  --rank=16 \
  --max_train_steps="${STEPS}" \
  --checkpointing_steps=400 \
  --seed="${SEED}" \
  --output_dir="${OUTPUT_DIR}" \
  --validation_prompt="raw recent-phone photo of ${INSTANCE_TOKEN} woman, natural face, realistic skin" \
  --num_validation_images=2 \
  --validation_epochs=5

sha256sum "${OUTPUT_DIR}/pytorch_lora_weights.safetensors"
