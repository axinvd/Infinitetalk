# Single-stage build: runtime with ComfyUI + custom nodes
# Models are downloaded at startup (not baked into the image) to keep image small
# Base includes: CUDA 12.8.1, cuDNN, Python 3.12, PyTorch 2.8.0, torchvision, torchaudio
FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System packages (ffmpeg for video, git-lfs for model downloads, libgl1 for OpenCV)
RUN apt-get update --yes && \
    apt-get install --yes --no-install-recommends ffmpeg git-lfs libgl1 && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Python dependencies (consolidated into one layer).
# xformers omitted: latest stable pulls torch 2.11 and silently uninstalls the
# base image's torch 2.8.0+cu128, breaking torchaudio/torchvision and the flash-attn
# wheel (which is torch-2.8-specific). ComfyUI runs with --use-sage-attention here,
# so xformers is not required.
# flash_attn: prebuilt wheel for Python 3.12 + CUDA 12.8 + PyTorch 2.8 (NO source compile).
# Filename includes both manylinux_2_24 and manylinux_2_28 tags — without the second
# tag the GitHub release URL 404s.
RUN pip install --no-cache-dir sageattention \
        misaki[en] "huggingface_hub[hf_transfer]" \
        runpod websocket-client librosa && \
    pip install --no-cache-dir \
        "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3+cu128torch2.8-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl"

WORKDIR /

# ComfyUI core. We keep comfyui-frontend-package — recent ComfyUI (>=0.20)
# checks for it during startup even in headless mode and exits with an error
# if missing. Workflow-templates and embedded-docs are still safe to strip.
RUN git clone https://github.com/comfyanonymous/ComfyUI.git && \
    cd /ComfyUI && \
    pip install --no-cache-dir -r requirements.txt && \
    pip uninstall -y comfyui-workflow-templates \
        comfyui-workflow-templates-core comfyui-workflow-templates-media-api \
        comfyui-workflow-templates-media-image comfyui-workflow-templates-media-other \
        comfyui-workflow-templates-media-video comfyui-embedded-docs 2>/dev/null || true

# Custom nodes (clone all, then install requirements)
RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/city96/ComfyUI-GGUF && \
    git clone https://github.com/kijai/ComfyUI-KJNodes && \
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite && \
    git clone https://github.com/orssorbit/ComfyUI-wanBlockswap && \
    git clone https://github.com/kijai/ComfyUI-MelBandRoFormer && \
    git clone https://github.com/kijai/ComfyUI-WanVideoWrapper && \
    cd ComfyUI-GGUF && pip install --no-cache-dir -r requirements.txt && \
    cd ../ComfyUI-KJNodes && pip install --no-cache-dir -r requirements.txt && \
    cd ../ComfyUI-VideoHelperSuite && pip install --no-cache-dir -r requirements.txt && \
    cd ../ComfyUI-MelBandRoFormer && pip install --no-cache-dir -r requirements.txt && \
    cd ../ComfyUI-WanVideoWrapper && pip install --no-cache-dir -r requirements.txt && \
    find /ComfyUI -name ".git" -type d -exec rm -rf {} + 2>/dev/null || true && \
    find /ComfyUI -name "*.pyc" -delete 2>/dev/null || true && \
    find /ComfyUI -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Handler files
COPY . .
RUN chmod +x /entrypoint.sh

ENV RUNPOD_PING_INTERVAL=3000
ENV RUNPOD_INIT_TIMEOUT=600

CMD ["/entrypoint.sh"]
