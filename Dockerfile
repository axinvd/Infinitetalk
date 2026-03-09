# Stage 1: Download models in parallel (runs concurrently with stage 2 via BuildKit)
FROM debian:bookworm-slim AS models
RUN apt-get update && apt-get install -y --no-install-recommends wget ca-certificates python3 python3-pip python3-venv && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /models/diffusion_models /models/loras /models/vae /models/text_encoders /models/clip_vision /models/transformers/TencentGameMate/chinese-wav2vec2-base
RUN python3 -m venv /opt/models-venv
RUN /opt/models-venv/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/models-venv/bin/pip install --no-cache-dir "huggingface_hub[hf_transfer]"
# Commented-out GGUF models preserved for reference
# wget -q https://huggingface.co/Kijai/WanVideo_comfy_GGUF/resolve/main/InfiniteTalk/Wan2_1-InfiniteTalk_Single_Q8.gguf -O /models/diffusion_models/Wan2_1-InfiniteTalk_Single_Q8.gguf
# wget -q https://huggingface.co/Kijai/WanVideo_comfy_GGUF/resolve/main/InfiniteTalk/Wan2_1-InfiniteTalk_Multi_Q8.gguf -O /models/diffusion_models/Wan2_1-InfiniteTalk_Multi_Q8.gguf
# wget -q https://huggingface.co/city96/Wan2.1-I2V-14B-480P-gguf/resolve/main/wan2.1-i2v-14b-480p-Q8_0.gguf -O /models/diffusion_models/wan2.1-i2v-14b-480p-Q8_0.gguf
RUN wget -q https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/InfiniteTalk/Wan2_1-InfiniteTalk-Single_fp8_e4m3fn_scaled_KJ.safetensors -O /models/diffusion_models/Wan2_1-InfiniteTalk-Single_fp8_e4m3fn_scaled_KJ.safetensors & \
    wget -q https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/InfiniteTalk/Wan2_1-InfiniteTalk-Multi_fp8_e4m3fn_scaled_KJ.safetensors -O /models/diffusion_models/Wan2_1-InfiniteTalk-Multi_fp8_e4m3fn_scaled_KJ.safetensors & \
    wget -q https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors -O /models/diffusion_models/Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors & \
    wget -q https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors -O /models/loras/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors & \
    wget -q https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1_VAE_bf16.safetensors -O /models/vae/Wan2_1_VAE_bf16.safetensors & \
    wget -q https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/umt5-xxl-enc-fp8_e4m3fn.safetensors -O /models/text_encoders/umt5-xxl-enc-fp8_e4m3fn.safetensors & \
    wget -q https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors -O /models/clip_vision/clip_vision_h.safetensors & \
    wget -q https://huggingface.co/Kijai/MelBandRoFormer_comfy/resolve/main/MelBandRoformer_fp16.safetensors -O /models/diffusion_models/MelBandRoformer_fp16.safetensors & \
    wait && \
    for f in \
      /models/diffusion_models/Wan2_1-InfiniteTalk-Single_fp8_e4m3fn_scaled_KJ.safetensors \
      /models/diffusion_models/Wan2_1-InfiniteTalk-Multi_fp8_e4m3fn_scaled_KJ.safetensors \
      /models/diffusion_models/Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors \
      /models/loras/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors \
      /models/vae/Wan2_1_VAE_bf16.safetensors \
      /models/text_encoders/umt5-xxl-enc-fp8_e4m3fn.safetensors \
      /models/clip_vision/clip_vision_h.safetensors \
      /models/diffusion_models/MelBandRoformer_fp16.safetensors; \
    do [ -s "$f" ] || { echo "FAILED: $f is missing or empty"; exit 1; }; done
RUN /opt/models-venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="TencentGameMate/chinese-wav2vec2-base",
    local_dir="/models/transformers/TencentGameMate/chinese-wav2vec2-base",
    revision="main",
)
PY
RUN test -s /models/transformers/TencentGameMate/chinese-wav2vec2-base/pytorch_model.bin || \
    (echo "FAILED: wav2vec model download missing" && exit 1)

# Stage 2: Build runtime with ComfyUI + custom nodes
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04 AS runtime

# Remove any third-party apt sources to avoid issues with expiring keys.
RUN rm -f /etc/apt/sources.list.d/*.list

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV SHELL=/bin/bash
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# System packages (from base.Dockerfile)
RUN apt-get update --yes && \
    apt-get upgrade --yes && \
    apt install --yes --no-install-recommends git wget curl bash libgl1 software-properties-common openssh-server nginx rsync ffmpeg git-lfs && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt install python3.10 python3.10-venv python3.10-distutils -y --no-install-recommends && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen

# Python 3.10 + pip
RUN ln -s /usr/bin/python3.10 /usr/bin/python && \
    rm /usr/bin/python3 && \
    ln -s /usr/bin/python3.10 /usr/bin/python3 && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

RUN pip install --no-cache-dir -U wheel setuptools packaging

# PyTorch + xformers (CUDA 12.8)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
RUN pip install --no-cache-dir xformers --index-url https://download.pytorch.org/whl/cu128

# flash_attn pre-built for CUDA 12.8 + PyTorch 2.10 + Python 3.10
# Source: https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/tag/v0.7.16
RUN pip install --no-cache-dir ninja psutil && \
    pip install --no-cache-dir "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3%2Bcu128torch2.10-cp310-cp310-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl"

# Additional deps from base image
RUN pip install --no-cache-dir misaki[en] packaging transformers==4.48.2

# Runtime deps
RUN pip install --no-cache-dir -U "huggingface_hub[hf_transfer]" runpod websocket-client librosa

WORKDIR /

RUN git clone https://github.com/comfyanonymous/ComfyUI.git && \
    cd /ComfyUI && \
    pip install --no-cache-dir -r requirements.txt

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/Comfy-Org/ComfyUI-Manager.git && \
    git clone https://github.com/city96/ComfyUI-GGUF && \
    git clone https://github.com/kijai/ComfyUI-KJNodes && \
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite && \
    git clone https://github.com/orssorbit/ComfyUI-wanBlockswap && \
    git clone https://github.com/kijai/ComfyUI-MelBandRoFormer && \
    git clone https://github.com/kijai/ComfyUI-WanVideoWrapper && \
    cd ComfyUI-Manager && pip install --no-cache-dir -r requirements.txt && \
    cd ../ComfyUI-GGUF && pip install --no-cache-dir -r requirements.txt && \
    cd ../ComfyUI-KJNodes && pip install --no-cache-dir -r requirements.txt && \
    cd ../ComfyUI-VideoHelperSuite && pip install --no-cache-dir -r requirements.txt && \
    cd ../ComfyUI-MelBandRoFormer && pip install --no-cache-dir -r requirements.txt && \
    cd ../ComfyUI-WanVideoWrapper && pip install --no-cache-dir -r requirements.txt && \
    find /ComfyUI -name ".git" -type d -exec rm -rf {} + 2>/dev/null || true

# Merge downloaded models from stage 1
COPY --from=models /models/ /ComfyUI/models/

COPY . .
RUN chmod +x /entrypoint.sh

ENV RUNPOD_PING_INTERVAL=3000

CMD ["/entrypoint.sh"]
