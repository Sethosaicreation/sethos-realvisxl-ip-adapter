FROM nvidia/cuda:12.8.1-base-ubuntu24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git python3 python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv \
    && python -m pip install --upgrade pip

WORKDIR /worker
COPY requirements-worker.txt /worker/requirements-worker.txt
RUN python -m pip install \
        torch==2.8.0 torchvision==0.23.0 \
        --index-url https://download.pytorch.org/whl/cu128 \
    && python -m pip install -r /worker/requirements-worker.txt

RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='h94/IP-Adapter', revision='018e402774aeeddd60609b4ecdb7e298259dc729', local_dir='/opt/models/ip-adapter', allow_patterns=['models/image_encoder/config.json','models/image_encoder/model.safetensors','sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors','sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors'])"

FROM nvidia/cuda:12.8.1-base-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH=/opt/venv/bin:$PATH \
    REALVIS_MODEL_ID=SG161222/RealVisXL_V5.0 \
    REALVIS_MODEL_REVISION=ac93e0dda1f6d448cae19bbfab8c5e720a5e48bc \
    IP_ADAPTER_MODEL_ID=h94/IP-Adapter \
    IP_ADAPTER_MODEL_REVISION=018e402774aeeddd60609b4ecdb7e298259dc729 \
    IP_ADAPTER_MODEL_DIR=/opt/models/ip-adapter

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /worker
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/models/ip-adapter /opt/models/ip-adapter

ENV HF_HOME=/runpod-volume/huggingface-cache \
    CHARACTER_LORA_ROOT=/runpod-volume/sethos-lora \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false

COPY schema.py media.py inference.py handler.py /worker/
RUN python -m py_compile /worker/schema.py /worker/media.py /worker/inference.py /worker/handler.py \
    && python -c "from diffusers import AutoPipelineForText2Image; from transformers import CLIPVisionModelWithProjection; from peft import PeftModel; assert AutoPipelineForText2Image and CLIPVisionModelWithProjection and PeftModel" \
    && test -f /opt/models/ip-adapter/sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors \
    && test -f /opt/models/ip-adapter/sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors

ENTRYPOINT []
CMD ["python", "-u", "/worker/handler.py"]
