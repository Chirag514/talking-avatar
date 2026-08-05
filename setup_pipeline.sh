#!/bin/bash
# setup_pipeline.sh
#
# One-time setup for a rented GPU pod (RunPod or similar). Run once per
# pod/volume -- everything it downloads goes to the persistent network
# volume (/workspace), so a pod STOP/restart doesn't require re-running
# this. If you TERMINATE (delete) the pod and volume together, you'll
# need to run this again on a fresh volume.
#
# Replaces the old setup_runpod.sh + setup_echomimic.sh pair. This
# pipeline now has a single animation stage (Ditto) -- LatentSync and
# EchoMimicV3 setup have both been removed. See the Trials document
# for why Ditto was selected.
#
# Usage:
#   chmod +x setup_pipeline.sh
#   ./setup_pipeline.sh
#
# Assumes: a pod with a persistent volume mounted at /workspace, an
# NVIDIA GPU, and CUDA already available in the base image.

set -e

echo "=== Path A Pipeline -- setup ==="

mkdir -p /workspace/models
mkdir -p /workspace/pipeline_runs
mkdir -p /workspace/hf_cache

# ---------------------------------------------------------------------
# 1. Main environment -- Python dependencies
# ---------------------------------------------------------------------
echo "--- Installing main-environment Python dependencies ---"

pip install -q --upgrade "transformers>=5.3.0"
pip install -q soundfile
pip install -q omnivoice --no-deps   # --no-deps: protects your pinned torch version, see stage2_voice_gen.py
pip install -q openai-whisper        # used by OmniVoice's built-in ref_audio auto-transcription
pip install -q openai                # for stage1_safety_gate.py's moderation API call
pip install -q groq                  # for stage7_overlays.py's callout-detection LLM call (optional feature)
pip install -q moviepy pillow        # for stage7_overlays.py's video compositing (optional feature)
pip install -q opencv-python         # for stage7_overlays.py's face-detection-based placement (optional feature)
pip install -q playwright            # for overlay_render.py's HTML->image rendering (optional feature, Stage 7)
playwright install --with-deps chromium 2>/dev/null || \
    echo "WARNING: 'playwright install chromium' failed -- Stage 7 overlay rendering will not work until this succeeds. Try running it manually (may need sudo for --with-deps)."

apt-get install -qq -y fonts-noto-core 2>/dev/null || echo "WARNING: fonts-noto-core install failed -- Hindi overlay text may not render."

echo "--- Installing ImageMagick + fixing its policy for MoviePy TextClip (Stage 7) ---"
apt-get install -qq -y imagemagick 2>/dev/null || echo "WARNING: imagemagick install failed -- Stage 7 text overlays will not render."
POLICY_PATH="/etc/ImageMagick-6/policy.xml"
if [ -f "$POLICY_PATH" ]; then
    sed -i 's/rights="none" pattern="@\*"/rights="read|write" pattern="@*"/' "$POLICY_PATH"
    sed -i 's/rights="none"/rights="read|write"/' "$POLICY_PATH" 2>/dev/null || true
else
    echo "WARNING: $POLICY_PATH not found -- if Stage 7 TextClip calls fail with a policy error, locate and edit your ImageMagick policy.xml manually."
fi

# ---------------------------------------------------------------------
# 2. Real-ESRGAN -- clone + weights to the persistent volume
# ---------------------------------------------------------------------
echo "--- Setting up Real-ESRGAN ---"

if [ ! -d "/workspace/models/Real-ESRGAN" ]; then
    git clone https://github.com/xinntao/Real-ESRGAN.git /workspace/models/Real-ESRGAN
    cd /workspace/models/Real-ESRGAN
    pip install -q -r requirements.txt
    python setup.py develop
    mkdir -p weights
    wget -q -O weights/RealESRGAN_x4plus.pth \
        https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
    cd -
else
    echo "Real-ESRGAN already present at /workspace/models/Real-ESRGAN, skipping clone."
fi

echo "--- Patching basicsr for current torchvision (Stage 5 fix) ---"
BASICSR_DEGRADATIONS=$(python3 -c "import basicsr, os; print(os.path.join(os.path.dirname(basicsr.__file__), 'data', 'degradations.py'))" 2>/dev/null || echo "")
if [ -n "$BASICSR_DEGRADATIONS" ] && [ -f "$BASICSR_DEGRADATIONS" ]; then
    if grep -q "torchvision.transforms.functional_tensor" "$BASICSR_DEGRADATIONS"; then
        sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms._functional_tensor import rgb_to_grayscale/' "$BASICSR_DEGRADATIONS"
        echo "Patched: $BASICSR_DEGRADATIONS"
    else
        echo "basicsr degradations.py already patched or uses a different import -- nothing to do."
    fi
else
    echo "basicsr not yet installed (installed above via Real-ESRGAN's requirements.txt) -- if this warning appears, check that step ran, then re-run this script."
fi

# ---------------------------------------------------------------------
# 3. Ditto -- dedicated conda environment + checkpoints
# ---------------------------------------------------------------------
# Ditto's own CODE lives in this repo already, at pipeline/ditto_talkinghead/
# (brought in-tree so this is a single codebase). Only its Python
# environment and its checkpoints (too large for git) need setting up here.
echo "--- Setting up Ditto conda environment ---"

DITTO_DIR="$(cd "$(dirname "$0")/pipeline/ditto_talkinghead" && pwd)"

if ! conda env list | grep -q "^ditto "; then
    conda env create -f "$DITTO_DIR/environment.yaml"
else
    echo "conda env 'ditto' already exists, skipping creation."
fi

# The environment.yaml's pip section includes tensorrt/tensorrt-libs/
# tensorrt-bindings==8.6.1, which failed to resolve during development
# (no matching wheel for this platform) and isn't needed anyway -- this
# pipeline uses Ditto's PyTorch checkpoints, not its TensorRT ones.
# `conda env create` will likely fail partway through the pip section;
# that's expected. If it does, finish the env manually:
#
#   conda activate ditto
#   pip install audioread==3.0.1 cffi==1.17.1 cuda-python==12.6.2.post1 \
#       cython==3.0.11 decorator==5.1.1 filetype==1.2.0 imageio==2.36.1 \
#       imageio-ffmpeg==0.5.1 joblib==1.4.2 lazy-loader==0.4 \
#       librosa==0.10.2.post1 llvmlite==0.43.0 msgpack==1.1.0 numba==0.60.0 \
#       nvidia-cublas-cu12==12.6.4.1 nvidia-cuda-runtime-cu12==12.6.77 \
#       nvidia-cudnn-cu12==9.6.0.74 opencv-python-headless==4.10.0.84 \
#       packaging==24.2 platformdirs==4.3.6 pooch==1.8.2 pycparser==2.22 \
#       scikit-image==0.25.0 scikit-learn==1.6.0 scipy==1.15.0 \
#       soundfile==0.13.0 soxr==0.5.0.post1 threadpoolctl==3.5.0 \
#       tifffile==2024.12.12 tqdm==4.67.1 polygraphy colored
#
# Additionally, three dependencies were needed at runtime but are NOT
# listed anywhere in Ditto's own requirements/environment.yaml (an
# upstream documentation gap, not something we did wrong):
conda run -n ditto pip install -q onnxruntime mediapipe einops || \
    echo "WARNING: could not install onnxruntime/mediapipe/einops into the ditto env automatically -- run manually: conda activate ditto && pip install onnxruntime mediapipe einops"

echo "--- Downloading Ditto checkpoints ---"
CHECKPOINTS_DIR="$DITTO_DIR/checkpoints"
if [ ! -d "$CHECKPOINTS_DIR" ] || [ -z "$(ls -A "$CHECKPOINTS_DIR" 2>/dev/null)" ]; then
    git lfs install
    git clone https://huggingface.co/digital-avatar/ditto-talkinghead "$CHECKPOINTS_DIR"
else
    echo "Ditto checkpoints already present at $CHECKPOINTS_DIR, skipping."
fi

# ---------------------------------------------------------------------
# 4. Sanity check
# ---------------------------------------------------------------------
echo "--- Verifying GPU visibility (main env) ---"
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"

echo ""
echo "=== Setup complete ==="
echo "Persistent volume / repo contents:"
echo "  /workspace/models/Real-ESRGAN         -- code + weights"
echo "  pipeline/ditto_talkinghead/           -- Ditto code (in-repo)"
echo "  pipeline/ditto_talkinghead/checkpoints/ -- Ditto weights (downloaded, not in git)"
echo "  /workspace/hf_cache                   -- OmniVoice weights land here on first run"
echo "  /workspace/pipeline_runs              -- final video + manifest synced here after each run"
echo ""
echo "Set OPENAI_API_KEY as an env var before running the pipeline (stage1_safety_gate.py needs it):"
echo "  export OPENAI_API_KEY=sk-..."
echo ""
echo "If using --overlays (Stage 7), also set GROQ_API_KEY:"
echo "  export GROQ_API_KEY=gsk_..."
echo ""
echo "To test individual stages standalone, use test_stages.sh -- e.g.:"
echo "  ./test_stages.sh voice script.txt reference_voice.wav --speed 1.4"
echo "  ./test_stages.sh ditto reference_image.png test_output/voice.wav"
echo ""
echo "Then run a full video with:"
echo "  conda run -n ditto python pipeline/run_pipeline.py \\"
echo "      script.txt reference_image.png reference_voice.wav output.mp4 \\"
echo "      --submitting_user_id user_123 --speed 1.4"
