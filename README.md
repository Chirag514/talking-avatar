# talking-avatar

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

This is a **single codebase** — Ditto's own code lives in-tree at
`pipeline/ditto_talkinghead/` rather than as a separately-managed
clone elsewhere on disk. Ditto was selected over EchoMimicV3,
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
export OPENAI_API_KEY=sk-...      # stage1_safety_gate.py -- not needed if using --skip_safety_gate
export GROQ_API_KEY=gsk_...       # stage7_overlays.py, only if using --overlays
```

## Running the full pipeline

```bash
conda run -n ditto python pipeline/run_pipeline.py \
    script.txt reference_image.png reference_voice.wav output.mp4 \
    --submitting_user_id user_123 --speed 1.4
```

Add `--overlays` to enable animated callouts, or `--no_restoration` to
skip the Real-ESRGAN upscale pass.

Add `--skip_safety_gate` to skip Stage 1 entirely (no `OPENAI_API_KEY`
needed) — dev/testing only. Content moderation is the only real check
Stage 1 performs today (the consent check is a permanent stub, see
`stage1_safety_gate.py`), so skipping it means the script text is not
checked against OpenAI's moderation API before generation.

## Testing individual stages

```bash
./test_stages.sh voice script.txt reference_voice.wav --speed 1.4
./test_stages.sh ditto reference_image.png test_output/cloned_voice.wav
./test_stages.sh restore test_output/ditto_test.mp4
./test_stages.sh export test_output/restored.mp4 test_output/cloned_voice.wav final.mp4
```

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
├── setup_pipeline.sh
├── test_stages.sh
└── test_output/                  # generated during testing, gitignored
```

> No sample reference photo/voice clip is included in this repo.
> Supply your own `reference_image.png` / `reference_voice.wav` when
> running the commands above, or use the assets under
> `pipeline/ditto_talkinghead/example/` to test Ditto in isolation.
