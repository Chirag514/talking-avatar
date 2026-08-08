# talking-avatar

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Chirag514/talking-avatar/blob/main/notebooks/talking_avatar_colab.ipynb)

Offline avatar video generation: a reference photo + a text script + a
reference voice clip in, a talking-avatar video out.

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

Ditto was selected over EchoMimicV3, LatentSync, JoyVASA/LivePortrait, MuseTalk,
Wav2Lip, Hallo2/3, and V-Express after evaluating each.

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

Installs main-environment dependencies, sets up Real-ESRGAN, creates
the `ditto` conda env, and downloads Ditto's checkpoints (not in git —
several GB). See the script's comments for the fixes baked into each
step.

```bash
export OPENAI_API_KEY=sk-...      # stage1_safety_gate.py
export GROQ_API_KEY=gsk_...       # stage7_overlays.py, only if using --overlays
```

## Running

This clones a real person's voice and face — `--i_confirm_consent` is
required, attesting the person in the reference files has agreed to
this use. See "Safety gate & consent" below for exactly what that does
and doesn't cover.

```bash
conda run -n ditto python pipeline/run_pipeline.py \
    script.txt reference_image.png reference_voice.wav output.mp4 \
    --submitting_user_id user_123 --speed 1.4 --i_confirm_consent
```

`--overlays` adds animated callouts. `--no_restoration` skips the
Real-ESRGAN upscale pass.

## Safety gate & consent

Stage 1 runs two checks before anything generates:

1. **Content moderation** — script text against OpenAI's moderation
   API. Checks what gets *said*, not who the voice/face belongs to.
2. **Consent attestation** — `--i_confirm_consent`, logged (who, file
   hashes, when). This is an explicit attestation, **not** identity or
   liveness verification — it can't confirm the reference files'
   subject actually consented, only that someone claimed so and it's
   recorded. See `check_consent()` in `stage1_safety_gate.py` for the
   known gap and what a real fix looks like.

Both must pass before Stage 2+ runs.

## Testing individual stages

```bash
./test_stages.sh voice script.txt reference_voice.wav --speed 1.4
./test_stages.sh ditto reference_image.png test_output/cloned_voice.wav
./test_stages.sh restore test_output/ditto_test.mp4
./test_stages.sh export test_output/restored.mp4 test_output/cloned_voice.wav final.mp4
```

## Testing on Google Colab

Confirmed working end-to-end on a Colab T4. Open
[`notebooks/talking_avatar_colab.ipynb`](notebooks/talking_avatar_colab.ipynb)
(badge above) and run top to bottom — includes the Colab-specific
environment setup and a troubleshooting section for the gotchas
already hit (conda's torch reporting no CUDA, `onnxruntime` needing
the GPU build, etc).

## Ditto tuning — what's baked in and why

`stage4c_ditto.py`'s defaults aren't Ditto's out-of-the-box behavior.
Six fixes, found via source tracing + iterative testing (full detail
in each fix's docstring):

1. **Lip-sync accuracy** — reduced default temporal smoothing, which
   was over-smoothing mouth-shape detail.
2. **Head motion** — Ditto predicts zero head pose from audio alone on
   a static image. Fixed with synthetic, audio-reactive sway.
3. **Cold-start artifact** — first ~15 frames show an unnatural mouth
   position; fixed via Ditto's own unused `fade_in` parameter.
4. **Expression lock** — Ditto hardcodes everything but lip/eye
   keypoints to the static source image. Fixed via a direct override
   of the model's internal blend-weight attributes post-`setup()`.
5. **Blink motion** — default blinks are abrupt; damped.
6. **`emo` parameter** — tested exhaustively, no visible effect on
   this checkpoint. Not used.

**Out of scope:** full-body/shoulder/hand motion — Ditto is
architecturally face-only (inherited from LivePortrait).

## Repository layout

```
talking-avatar/
├── pipeline/
│   ├── run_pipeline.py           # single entrypoint
│   ├── stages/                   # each pipeline stage
│   ├── ditto_talkinghead/        # Ditto's code, in-tree (checkpoints/ downloaded, not in git)
│   └── utils/
├── notebooks/
│   └── talking_avatar_colab.ipynb
├── setup_pipeline.sh
├── test_stages.sh
```

No sample reference photo/voice clip is included — supply your own,
or use `pipeline/ditto_talkinghead/example/` to test Ditto in
isolation.