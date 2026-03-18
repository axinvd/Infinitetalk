#!/bin/bash
# Pod setup script for InfiniteTalk on A100 80GB
# Replicates the Dockerfile environment for testing, then builds Docker image
set -e

echo "=== InfiniteTalk Pod Setup ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'checking...')"

# --- System deps ---
echo "=== Installing system packages ==="
apt-get update --yes
apt install --yes --no-install-recommends git wget curl bash libgl1 software-properties-common ffmpeg git-lfs
add-apt-repository ppa:deadsnakes/ppa -y
apt install python3.10 python3.10-venv python3.10-distutils -y --no-install-recommends
apt-get clean
rm -rf /var/lib/apt/lists/*

# --- Python 3.10 as default ---
echo "=== Setting up Python 3.10 ==="
ln -sf /usr/bin/python3.10 /usr/bin/python
rm -f /usr/bin/python3
ln -sf /usr/bin/python3.10 /usr/bin/python3
curl -sS https://bootstrap.pypa.io/get-pip.py | python

# --- PyTorch + ML deps (CUDA 12.4 on this pod) ---
echo "=== Installing PyTorch + ML dependencies ==="
pip install --no-cache-dir torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu124

echo "=== Installing remaining pip packages ==="
pip install --no-cache-dir -U wheel setuptools packaging ninja psutil \
    misaki[en] "huggingface_hub[hf_transfer]" runpod websocket-client librosa sageattention

# flash_attn for CUDA 12.4 + Python 3.10
pip install --no-cache-dir flash-attn --no-build-isolation

# --- ComfyUI ---
echo "=== Setting up ComfyUI ==="
if [ ! -d "/ComfyUI" ]; then
    cd /
    git clone https://github.com/comfyanonymous/ComfyUI.git
    cd /ComfyUI
    pip install --no-cache-dir -r requirements.txt
    pip uninstall -y comfyui-frontend-package comfyui-workflow-templates \
        comfyui-workflow-templates-core comfyui-workflow-templates-media-api \
        comfyui-workflow-templates-media-image comfyui-workflow-templates-media-other \
        comfyui-workflow-templates-media-video comfyui-embedded-docs 2>/dev/null || true
fi

# --- Custom nodes ---
echo "=== Installing custom nodes ==="
cd /ComfyUI/custom_nodes
for repo in \
    "https://github.com/city96/ComfyUI-GGUF" \
    "https://github.com/kijai/ComfyUI-KJNodes" \
    "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite" \
    "https://github.com/orssorbit/ComfyUI-wanBlockswap" \
    "https://github.com/kijai/ComfyUI-MelBandRoFormer" \
    "https://github.com/kijai/ComfyUI-WanVideoWrapper"; do
    dirname=$(basename "$repo")
    if [ ! -d "$dirname" ]; then
        git clone "$repo"
        cd "$dirname"
        [ -f requirements.txt ] && pip install --no-cache-dir -r requirements.txt
        cd ..
    fi
done

# --- Models (reuses entrypoint.sh logic) ---
echo "=== Downloading models ==="
MODEL_DIR="/runpod-volume/models"
SENTINEL=".models_ready"

if [ -f "$MODEL_DIR/$SENTINEL" ]; then
    echo "Models already present on volume."
else
    mkdir -p "$MODEL_DIR/diffusion_models" "$MODEL_DIR/loras" "$MODEL_DIR/vae" \
             "$MODEL_DIR/text_encoders" "$MODEL_DIR/clip_vision" \
             "$MODEL_DIR/transformers/TencentGameMate"

    echo "Downloading models (~33GB)..."
    wget -q https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/InfiniteTalk/Wan2_1-InfiniteTalk-Single_fp8_e4m3fn_scaled_KJ.safetensors \
         -O "$MODEL_DIR/diffusion_models/Wan2_1-InfiniteTalk-Single_fp8_e4m3fn_scaled_KJ.safetensors" &
    wget -q https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/InfiniteTalk/Wan2_1-InfiniteTalk-Multi_fp8_e4m3fn_scaled_KJ.safetensors \
         -O "$MODEL_DIR/diffusion_models/Wan2_1-InfiniteTalk-Multi_fp8_e4m3fn_scaled_KJ.safetensors" &
    wget -q https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors \
         -O "$MODEL_DIR/diffusion_models/Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors" &
    wget -q https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors \
         -O "$MODEL_DIR/loras/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors" &
    wget -q https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1_VAE_bf16.safetensors \
         -O "$MODEL_DIR/vae/Wan2_1_VAE_bf16.safetensors" &
    wget -q https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/umt5-xxl-enc-fp8_e4m3fn.safetensors \
         -O "$MODEL_DIR/text_encoders/umt5-xxl-enc-fp8_e4m3fn.safetensors" &
    wget -q https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors \
         -O "$MODEL_DIR/clip_vision/clip_vision_h.safetensors" &
    wget -q https://huggingface.co/Kijai/MelBandRoFormer_comfy/resolve/main/MelBandRoformer_fp16.safetensors \
         -O "$MODEL_DIR/diffusion_models/MelBandRoformer_fp16.safetensors" &
    wait

    for f in \
        "$MODEL_DIR/diffusion_models/Wan2_1-InfiniteTalk-Single_fp8_e4m3fn_scaled_KJ.safetensors" \
        "$MODEL_DIR/diffusion_models/Wan2_1-InfiniteTalk-Multi_fp8_e4m3fn_scaled_KJ.safetensors" \
        "$MODEL_DIR/diffusion_models/Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors" \
        "$MODEL_DIR/loras/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors" \
        "$MODEL_DIR/vae/Wan2_1_VAE_bf16.safetensors" \
        "$MODEL_DIR/text_encoders/umt5-xxl-enc-fp8_e4m3fn.safetensors" \
        "$MODEL_DIR/clip_vision/clip_vision_h.safetensors" \
        "$MODEL_DIR/diffusion_models/MelBandRoformer_fp16.safetensors"; do
        [ -s "$f" ] || { echo "FAILED: $f is missing or empty"; exit 1; }
    done

    python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='TencentGameMate/chinese-wav2vec2-base',
    local_dir='$MODEL_DIR/transformers/TencentGameMate/chinese-wav2vec2-base',
    revision='main',
)
"
    touch "$MODEL_DIR/$SENTINEL"
    echo "Model download complete."
fi

# Symlink models into ComfyUI
for subdir in diffusion_models loras vae text_encoders clip_vision transformers; do
    rm -rf "/ComfyUI/models/$subdir"
    ln -sf "$MODEL_DIR/$subdir" "/ComfyUI/models/$subdir"
done
echo "Model symlinks created."

# --- Copy handler + workflows ---
echo "=== Copying handler and workflow files ==="
# These should be uploaded to /workspace/ first via runpodctl send or scp
if [ -d "/workspace/Infinitetalk_Runpod_hub" ]; then
    cp /workspace/Infinitetalk_Runpod_hub/handler.py /handler.py
    cp /workspace/Infinitetalk_Runpod_hub/I2V_single.json /I2V_single.json
    cp /workspace/Infinitetalk_Runpod_hub/I2V_multi.json /I2V_multi.json
    cp /workspace/Infinitetalk_Runpod_hub/V2V_single.json /V2V_single.json
    cp /workspace/Infinitetalk_Runpod_hub/V2V_multi.json /V2V_multi.json
    cp /workspace/Infinitetalk_Runpod_hub/entrypoint.sh /entrypoint.sh
    chmod +x /entrypoint.sh
    echo "Handler and workflows copied."
else
    echo "WARNING: /workspace/Infinitetalk_Runpod_hub not found."
    echo "Upload files first: scp -r Infinitetalk_Runpod_hub/ root@<pod>:/workspace/"
fi

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Next steps:"
echo "  1. Start ComfyUI:   python /ComfyUI/main.py --listen --use-sage-attention &"
echo "  2. Wait for ready:   curl -s http://127.0.0.1:8188/"
echo "  3. Start handler:    python /handler.py"
echo "  4. Test locally:     (from your machine) uv run client.py talk ..."
echo ""
echo "To build Docker image:"
echo "  docker build -t bbbasddaaa/talktalk:v1.0.0 -f /workspace/Infinitetalk_Runpod_hub/Dockerfile /workspace/Infinitetalk_Runpod_hub/"
echo "  docker login -u bbbasddaaa"
echo "  docker push bbbasddaaa/talktalk:v1.0.0"
