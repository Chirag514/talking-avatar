# talking-avatar

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Chirag514/talking-avatar/blob/main/notebooks/talking_avatar_colab.ipynb)

Fully offline avatar video generation: a reference photo + a text
script + a reference voice clip in, a talking-avatar video out.

```
script.txt + reference_voice.wav  ──►  OmniVoice  ──►  cloned_voice.wav
                                                            │
reference_image.png  ─────────────────────────────────────►│
                                                            ▼
                                                          Ditto
                                                            │
                                                            ▼
                                              (optional) Real-ESRGAN restoration
                                                            │
                                                            ▼
                                              (optional) animated overlays
                                                            │
                                                            ▼
                                                      final .mp4 export
```

Ditto was selected over EchoMimicV3,
LatentSync, JoyVASA/LivePortrait, MuseTalk, Wav2Lip, Hallo2/3, and
V-Express after evaluating each; see the fixes below for what it
took to get production-quality output out of it.

## Pipeline stages

| # | Stage | File | Environment |
|---|---|---|---|
| 1 | Safety gate (moderation + consent) | `pipeline/stages/stage1_safety_gate.py` | main |
| 2 | Voice cloning (OmniVoice) | `pipeline/stages/stage2_voice_gen.py` | main |
| 3 | Animation (Ditto) | `pipeline/stages/stage4c_ditto.py` | `ditto` conda env |
| 4 | Restoration (Real-ESRGAN, optional) | `pipeline/stages/stage5_restoration.py` | main |
| 5 | Overlay compositing (optional) | `pipeline/stages/stage7_overlays.py` | main |
| 6 | Export | `pipeline/stages/stage6_export.py` | main |

## Setup

```bash
chmod +x setup_pipeline.sh
./setup_pipeline.sh
```

This installs main-environment dependencies, sets up Real-ESRGAN,
creates the dedicated `ditto` conda environment, and downloads Ditto's
checkpoints (not stored in git — several GB). See the script itself
for what each step does and why; several steps encode real fixes found
during development (basicsr/torchvision patch, ImageMagick policy fix,
missing-dependency workarounds for Ditto's PyTorch checkpoint path).

Set required API keys before running:
```bash
export OPENAI_API_KEY=sk-...      # stage1_safety_gate.py
export GROQ_API_KEY=gsk_...       # stage7_overlays.py, only if using --overlays
```

## Running the full pipeline

This pipeline clones a real person's voice and face. Running it
requires an explicit consent attestation — `--i_confirm_consent` —
confirming the person in the reference photo/voice clip has actually
agreed to this use. See "Safety gate & consent" below for exactly
what this does and doesn't cover.

```bash
conda run -n ditto python pipeline/run_pipeline.py \
    script.txt reference_image.png reference_voice.wav output.mp4 \
    --submitting_user_id user_123 --speed 1.4 --i_confirm_consent
```

Add `--overlays` to enable animated callouts, or `--no_restoration` to
skip the Real-ESRGAN upscale pass.

## Safety gate & consent

Stage 1 (`stage1_safety_gate.py`) runs two independent checks before
any generation happens:

1. **Content moderation** — script text is checked against OpenAI's
   moderation API. Catches policy-violating things to *say*; doesn't
   know or care who the voice/face belongs to.
2. **Consent attestation** — requires `--i_confirm_consent` to be
   passed explicitly. This is a real, required, logged attestation
   (who attested, hashes of the reference files, when) — **not**
   identity or liveness verification. It cannot confirm the person in
   the reference files is who `--submitting_user_id` claims, or that
   they actually agreed. It only ensures a real "yes, I attest" signal
   is required and recorded, instead of silently assumed. Treat this
   as a floor, not a finished consent system — a self-upload match
   and/or liveness check is the real fix, and is a known gap, not a
   hidden one (see `check_consent()`'s docstring in
   `stage1_safety_gate.py`).

Both checks should pass before Stages 2+ run.


## Testing individual stages

```bash
./test_stages.sh voice script.txt reference_voice.wav --speed 1.4
./test_stages.sh ditto reference_image.png test_output/cloned_voice.wav
./test_stages.sh restore test_output/ditto_test.mp4
./test_stages.sh export test_output/restored.mp4 test_output/cloned_voice.wav final.mp4
```

## Testing on Google Colab

CONFIRMED working end-to-end (voice + Ditto animation) on a Colab T4,
as of 2026-08-08. Colab isn't conda-native and isn't persistent across
sessions, so this differs from the RunPod flow above in a few
specific, necessary ways.

**Fastest path:** open [`notebooks/talking_avatar_colab.ipynb`](notebooks/talking_avatar_colab.ipynb)
directly in Colab (badge at the top of this README) and run the cells
top to bottom — it's the same sequence below, already broken into
cells with a troubleshooting section at the end.

The full sequence, for reference / if you're building your own
notebook:

```bash
# 1. Get conda onto the instance
pip install -q condacolab
python -c "import condacolab; condacolab.install()"
# runtime restarts automatically here -- re-run cells below after it does

# 2. Mount Drive so downloaded weights survive session resets
python -c "
from google.colab import drive
drive.mount('/content/drive')
import os
os.makedirs('/content/drive/MyDrive/talking_avatar_models', exist_ok=True)
"

# 3. Clone the repo, main-env deps (same as setup_pipeline.sh section 1)
git clone https://github.com/Chirag514/talking-avatar.git
cd talking-avatar
pip install -q --upgrade "transformers>=5.3.0" soundfile
pip install -q omnivoice --no-deps
pip install -q torchaudio  # unpinned -- matches Colab's newer torch via stable ABI, see stage2_voice_gen.py
pip install -q accelerate gradio librosa pydub tensorboardx webdataset  # omnivoice's other --no-deps misses
pip install -q openai-whisper openai groq moviepy pillow opencv-python playwright
playwright install --with-deps chromium
apt-get install -qq -y fonts-noto-core imagemagick

# 4. Real-ESRGAN, on Drive so it isn't re-downloaded every session
[ ! -d "/content/drive/MyDrive/talking_avatar_models/Real-ESRGAN" ] && \
  git clone https://github.com/xinntao/Real-ESRGAN.git /content/drive/MyDrive/talking_avatar_models/Real-ESRGAN
cd /content/drive/MyDrive/talking_avatar_models/Real-ESRGAN
pip install -q -r requirements.txt && python setup.py develop
cd /content/talking-avatar

# 5. Ditto's env -- use environment_colab.yaml, NOT environment.yaml.
#    The full environment.yaml (with tensorrt pins) fails to resolve
#    on Colab entirely; environment_colab.yaml is a minimal subset
#    that CONFIRMED succeeds.
conda env create -f pipeline/ditto_talkinghead/environment_colab.yaml

# 6. CONFIRMED NECESSARY: conda's pytorch reports CUDA unavailable on
#    Colab (driver/build mismatch) -- force-reinstall via pip against
#    Colab's actual CUDA build:
conda run -n ditto pip uninstall -y torch torchvision torchaudio
conda run -n ditto pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121
# verify: should print CUDA available: True
conda run -n ditto python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# 7. Remaining pip-only packages environment.yaml would normally cover
#    (see setup_pipeline.sh's comment block for the full pinned list)
conda run -n ditto pip install -q audioread==3.0.1 cffi==1.17.1 cuda-python==12.6.2.post1 \
    cython==3.0.11 decorator==5.1.1 filetype==1.2.0 imageio==2.36.1 \
    imageio-ffmpeg==0.5.1 joblib==1.4.2 lazy-loader==0.4 \
    librosa==0.10.2.post1 llvmlite==0.43.0 msgpack==1.1.0 numba==0.60.0 \
    nvidia-cublas-cu12==12.6.4.1 nvidia-cuda-runtime-cu12==12.6.77 \
    nvidia-cudnn-cu12==9.6.0.74 opencv-python-headless==4.10.0.84 \
    packaging==24.2 platformdirs==4.3.6 pooch==1.8.2 pycparser==2.22 \
    scikit-image==0.25.0 scikit-learn==1.6.0 scipy==1.15.0 \
    soundfile==0.13.0 soxr==0.5.0.post1 threadpoolctl==3.5.0 \
    tifffile==2024.12.12 tqdm==4.67.1 polygraphy colored
conda run -n ditto pip install -q einops

# 8. CONFIRMED NECESSARY: plain `onnxruntime` is CPU-only. Swap for GPU:
conda run -n ditto pip uninstall -y onnxruntime
conda run -n ditto pip install onnxruntime-gpu
conda run -n ditto pip install -q mediapipe

# 9. Ditto checkpoints
git lfs install
[ ! -d "pipeline/ditto_talkinghead/checkpoints" ] && \
  git clone https://huggingface.co/digital-avatar/ditto-talkinghead pipeline/ditto_talkinghead/checkpoints

# 10. Test
conda run -n ditto python -c "import filetype, pyximport, onnxruntime, mediapipe, einops; print('all present')"
conda run -n ditto bash test_stages.sh ditto reference_image.png test_output/cloned_voice.wav
```

A `pyximport` import error means `cython` didn't actually install in
step 7 (it's bundled inside the `cython` package, not separate).

If `conda run -n ditto ...` ever fails with
`DirectoryNotACondaEnvironmentError`, the env creation in step 5 failed
early and left a broken stub directory -- `rm -rf /usr/local/envs/ditto`
and retry step 5, watching the full output (`2>&1 | tail -100`) rather
than assuming it succeeded.


## Ditto tuning — what's baked in and why

`stage4c_ditto.py`'s defaults are not Ditto's out-of-the-box behavior.
Six separate fixes were found through direct source-code tracing and
validated through iterative testing — full root-cause detail is in
each fix's own docstring in that file:

1. **Lip-sync accuracy** — reduced default temporal smoothing
   (`smo_k_d=1, smo_k_s=5`), which was over-smoothing fine mouth-shape
   detail.
2. **Head motion** — Ditto predicts zero head pose from audio alone for
   a static image (a structural limitation, not a bug). Fixed with
   synthetic, audio-reactive head-pose sway.
3. **Cold-start artifact** — the first ~15 frames show an unnatural
   mouth position; fixed via Ditto's own (previously unused) `fade_in`
   parameter.
4. **Expression lock** — Ditto hardcodes generated expression to only
   lip + eye keypoints; every other keypoint (eyebrows, cheeks, nose)
   is locked to the static source image with no exposed config flag.
   Fixed via a direct, partial override of the model's internal
   blend-weight attributes after `setup()`.
5. **Blink motion** — default blinks are abrupt; fixed by damping the
   blink displacement array.
6. **`emo` (emotion) parameter** — tested exhaustively, confirmed to
   have no visible effect in this checkpoint. Not used.

**Confirmed out of scope:** full-body / shoulder / hand motion. Ditto
is architecturally face-only (inherited from LivePortrait) — no
config change can add this.

## Repository layout

```
talking-avatar/
├── pipeline/
│   ├── run_pipeline.py           # single entrypoint
│   ├── stages/                   # each pipeline stage
│   ├── ditto_talkinghead/        # Ditto's code, in-tree (checkpoints/ downloaded, not in git)
│   └── utils/
├── notebooks/
│   └── talking_avatar_colab.ipynb  # runnable Colab setup + testing notebook
├── setup_pipeline.sh
├── test_stages.sh
└── test_output/                  # generated during testing, gitignored
```

> No sample reference photo/voice clip is included in this repo.
> Supply your own `reference_image.png` / `reference_voice.wav` when
> running the commands above, or use the assets under
> `pipeline/ditto_talkinghead/example/` to test Ditto in isolation.
