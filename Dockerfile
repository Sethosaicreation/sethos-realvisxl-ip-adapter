FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    REALVIS_MODEL_ID=SG161222/RealVisXL_V5.0 \
    REALVIS_MODEL_REVISION=ac93e0dda1f6d448cae19bbfab8c5e720a5e48bc \
    IP_ADAPTER_MODEL_ID=h94/IP-Adapter \
    IP_ADAPTER_MODEL_REVISION=018e402774aeeddd60609b4ecdb7e298259dc729 \
    IP_ADAPTER_MODEL_DIR=/opt/models/ip-adapter

RUN apt-get update \
    && apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /worker
COPY requirements-worker.txt /worker/requirements-worker.txt
RUN python -m pip install -r /worker/requirements-worker.txt

RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='h94/IP-Adapter', revision='018e402774aeeddd60609b4ecdb7e298259dc729', local_dir='/opt/models/ip-adapter', allow_patterns=['models/image_encoder/**','sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors','sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors'])"

ENV HF_HOME=/runpod-volume/huggingface-cache \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false

COPY schema.py media.py inference.py handler.py /worker/
RUN python -m py_compile /worker/schema.py /worker/media.py /worker/inference.py /worker/handler.py \
    && python -c "from diffusers import AutoPipelineForText2Image; from transformers import CLIPVisionModelWithProjection; assert AutoPipelineForText2Image and CLIPVisionModelWithProjection" \
    && test -f /opt/models/ip-adapter/sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors \
    && test -f /opt/models/ip-adapter/sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors

ENTRYPOINT []
CMD ["python", "-u", "/worker/handler.py"]
