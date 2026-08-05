"""
stage4c_ditto.py

Stage 4C -- Image-driven animation via Ditto (antgroup/ditto-talkinghead)
Takes a REFERENCE IMAGE + an audio clip, and generates a talking-avatar
video using Ditto's motion-space diffusion approach.

THIS IS THE SELECTED PRODUCTION ANIMATION STAGE. See the Trials
document for the full comparison against EchoMimicV3, LatentSync,
JoyVASA/LivePortrait, Hallo2/3, MuseTalk, Wav2Lip, and V-Express.
Ditto won on lip-sync accuracy, generation speed, and VRAM headroom.

This file bakes in every fix found through extensive manual tuning
(see the Trials document, Section 3.5, for the full root-cause story
behind each one). None of these are default Ditto behavior -- every
one required either a parameter override or a direct monkey-patch of
internal model-stitching attributes:

  1. Lip-sync quality: default temporal smoothing (smo_k_d=3, smo_k_s=13)
     over-smooths fine mouth-shape detail. Fixed via smo_k_d=1, smo_k_s=5.

  2. Head motion: Ditto's audio-driven diffusion model (LMDM) does not
     predict head pose from audio for a static image source -- confirmed
     as a structural limitation (LivePortrait-inherited architecture
     expects head pose from a driving video, not audio). Fixed by
     injecting synthetic head-pose sway via ctrl_info, with amplitude
     modulated by the audio's RMS energy envelope so motion intensifies
     during emphasized speech and settles during pauses.

  3. Cold-start artifact: the first ~15 frames show unnatural mouth
     position before the model stabilizes. Fixed using the (previously
     unused) built-in fade_in parameter.

  4. Expression lock: motion_stitch.py hardcodes audio/model-driven
     expression to only 6 lip + 5 eye keypoints (of 21 total); every
     other keypoint (eyebrows, cheeks, nose, forehead) is forced to the
     static source image's values, unconditionally, with no exposed
     config flag. Confirmed via source trace -- not fixable via any
     setup() kwarg. Fixed by directly overriding the model's internal
     fix_exp_a1/a2/a3 blend-weight attributes after setup() completes,
     partially unlocking the non-lip/non-eye keypoints (alpha tested
     and confirmed working at 0.4-0.6; higher values not yet fully
     explored for artifact onset).

  5. Blink motion: default blinks apply a fixed 15-frame keyframe
     sequence added at full strength directly on top of the source
     expression, with no damping -- reads as an abrupt/forceful close.
     Fixed by scaling delta_eye_arr by 0.5.

  6. emo (emotion-conditioning) parameter: tested across all 8 index
     values, including with the expression lock relaxed. Confirmed via
     code trace that the signal reaches the model, but produced no
     visible output difference in any configuration tested. Treated as
     a confirmed dead end -- not used here.

CONFIRMED OUT OF SCOPE: full-body/shoulder/hand motion. Ditto's face
detector, 21-point landmark representation, and warping/rendering
network are face-only by design (inherited from LivePortrait). No
config change or monkey-patch can add body motion -- see the Trials
document, Section 3.5.7.
"""

import os
import sys
import math
import subprocess
import time
from pathlib import Path

import numpy as np

# Ditto's own code (core/, inference.py, stream_pipeline_offline.py,
# environment.yaml, etc.) lives inside this same repo at
# pipeline/ditto_talkinghead/ -- see that folder's own README.md for
# the upstream project's original documentation.
#
# Checkpoints are NOT included in this repo. Download them per
# pipeline/ditto_talkinghead/README.md's "Download Checkpoints"
# section, or see this project's top-level README.
DITTO_REPO_PATH = os.environ.get(
    "DITTO_REPO_PATH",
    str(Path(__file__).resolve().parent.parent / "ditto_talkinghead"),
)

DITTO_CONDA_ENV = os.environ.get("DITTO_CONDA_ENV", "ditto")

DITTO_DATA_ROOT = os.environ.get(
    "DITTO_DATA_ROOT", f"{DITTO_REPO_PATH}/checkpoints/ditto_pytorch"
)
DITTO_CFG_PKL = os.environ.get(
    "DITTO_CFG_PKL", f"{DITTO_REPO_PATH}/checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl"
)

DEFAULT_SMO_K_D = 1
DEFAULT_SMO_K_S = 5
DEFAULT_FADE_IN_FRAMES = 15
DEFAULT_BLINK_SCALE = 0.5
DEFAULT_EXPRESSION_ALPHA = 0.6

SWAY_PITCH_AMPLITUDE_DEG = 3.0
SWAY_PITCH_PERIOD_S = 4.0
SWAY_YAW_AMPLITUDE_DEG = 4.0
SWAY_YAW_PERIOD_S = 6.5
SWAY_YAW_PHASE = 1.0
SWAY_ROLL_AMPLITUDE_DEG = 1.5
SWAY_ROLL_PERIOD_S = 9.0
SWAY_ROLL_PHASE = 2.0
SWAY_ENERGY_FLOOR = 0.3

_LIP_KEYPOINTS = [6, 12, 14, 17, 19, 20]
_EYE_KEYPOINTS = [11, 13, 15, 16, 18]


def _build_audio_reactive_ctrl_info(audio_path, num_frames, fps=25):
    """Per-frame ctrl_info injecting synthetic head-pose sway, amplitude
    modulated by the audio's smoothed RMS energy envelope (fix #2)."""
    import librosa

    audio, sr = librosa.core.load(audio_path, sr=16000)
    hop_length = int(sr / fps)
    rms = librosa.feature.rms(y=audio, frame_length=hop_length * 2, hop_length=hop_length)[0]
    rms_smooth = np.convolve(rms, np.ones(5) / 5, mode="same")
    rms_norm = rms_smooth / (rms_smooth.max() + 1e-8)
    rms_norm = SWAY_ENERGY_FLOOR + (1 - SWAY_ENERGY_FLOOR) * rms_norm
    rms_norm = np.pad(rms_norm, (0, max(0, num_frames - len(rms_norm))), mode="edge")[:num_frames]

    ctrl_info = {}
    for fid in range(num_frames):
        t = fid / fps
        env = float(rms_norm[fid])
        ctrl_info[fid] = {
            "delta_pitch": env * SWAY_PITCH_AMPLITUDE_DEG * math.sin(2 * math.pi * t / SWAY_PITCH_PERIOD_S),
            "delta_yaw": env * SWAY_YAW_AMPLITUDE_DEG * math.sin(2 * math.pi * t / SWAY_YAW_PERIOD_S + SWAY_YAW_PHASE),
            "delta_roll": env * SWAY_ROLL_AMPLITUDE_DEG * math.sin(2 * math.pi * t / SWAY_ROLL_PERIOD_S + SWAY_ROLL_PHASE),
        }
    return ctrl_info


_EXPECTED_KEYPOINT_COUNT = 21  # what _LIP_KEYPOINTS/_EYE_KEYPOINTS indices assume


def _validate_ditto_internals(sdk):
    """
    Fails loudly, with a specific diagnosis, if Ditto's internal
    attribute names or shapes no longer match what fixes #4/#5 assume.

    Call this once right after sdk.setup(), before any monkey-patch
    below. Without it: if fix_exp_a1/a2 change shape (e.g. Ditto moves
    to a different keypoint count) but total size still happens to be
    divisible by 3, .reshape(21, 3) SUCCEEDS with no error -- it just
    silently blends the wrong keypoints. A crash here is much easier
    to catch than a video where eyebrows move like they shouldn't.
    """
    ms = sdk.motion_stitch
    required = ["fix_exp_a1", "fix_exp_a2", "delta_eye_arr"]
    missing = [name for name in required if not hasattr(ms, name)]
    if missing:
        raise RuntimeError(
            f"Ditto's motion_stitch object is missing attribute(s) {missing}, "
            f"which _apply_expression_unlock()/_apply_blink_softening() in this "
            f"file depend on directly. Your installed Ditto checkpoint/code no "
            f"longer matches what this file was written against. Check "
            f"pipeline/ditto_talkinghead/core/atomic_components/motion_stitch.py "
            f"for the current attribute names and update this file to match "
            f"before trusting any output from this run."
        )

    for name in ("fix_exp_a1", "fix_exp_a2"):
        arr = getattr(ms, name)
        total = int(arr.size) if hasattr(arr, "size") else len(arr)
        expected_total = _EXPECTED_KEYPOINT_COUNT * 3
        if total != expected_total:
            raise RuntimeError(
                f"Ditto's motion_stitch.{name} has {total} value(s), expected "
                f"{expected_total} ({_EXPECTED_KEYPOINT_COUNT} keypoints x 3 "
                f"[pitch/yaw/roll or xyz]). This means Ditto's keypoint layout "
                f"has changed -- _LIP_KEYPOINTS/_EYE_KEYPOINTS in this file "
                f"index into what they assume is a {_EXPECTED_KEYPOINT_COUNT}-"
                f"keypoint array. Proceeding would apply fix #4's expression "
                f"unlock to the WRONG keypoints with no crash to warn you. "
                f"Update _EXPECTED_KEYPOINT_COUNT, _LIP_KEYPOINTS, and "
                f"_EYE_KEYPOINTS to match the new layout first."
            )


def _apply_expression_unlock(sdk, alpha):
    """Partially relaxes Ditto's hardcoded expression lock (fix #4).
    Leaves lip/eye keypoints untouched (already proven working); only
    the remaining ~10 keypoints get blended toward the model's own
    prediction."""
    locked = sorted(set(range(21)) - set(_LIP_KEYPOINTS) - set(_EYE_KEYPOINTS))

    a1 = sdk.motion_stitch.fix_exp_a1.copy().reshape(21, 3)
    a2 = sdk.motion_stitch.fix_exp_a2.copy().reshape(21, 3)
    a1[locked] = alpha
    a2[locked] = 1 - alpha
    sdk.motion_stitch.fix_exp_a1 = a1.reshape(1, -1)
    sdk.motion_stitch.fix_exp_a2 = a2.reshape(1, -1)


def _apply_blink_softening(sdk, blink_scale):
    """Softens the abrupt default blink motion (fix #5)."""
    sdk.motion_stitch.delta_eye_arr = sdk.motion_stitch.delta_eye_arr * blink_scale


def run_ditto(reference_image_path, audio_path, output_video_path,
              smo_k_d=DEFAULT_SMO_K_D,
              smo_k_s=DEFAULT_SMO_K_S,
              fade_in_frames=DEFAULT_FADE_IN_FRAMES,
              blink_scale=DEFAULT_BLINK_SCALE,
              expression_alpha=DEFAULT_EXPRESSION_ALPHA,
              enable_audio_reactive_sway=True):
    """
    Runs Ditto image-to-video inference with every validated fix applied.

    NOTE: this calls into Ditto's own StreamSDK directly (not via
    subprocess to inference.py), since the fixes above require
    reaching into the SDK object after setup() but before generation.
    Run this stage under DITTO_CONDA_ENV, not the main pipeline env --
    or call run_ditto_subprocess() from a different env instead.
    """
    reference_image_path = Path(reference_image_path).resolve()
    audio_path = Path(audio_path).resolve()
    output_video_path = Path(output_video_path).resolve()
    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, DITTO_REPO_PATH)
    from stream_pipeline_offline import StreamSDK
    import librosa

    audio, sr = librosa.core.load(str(audio_path), sr=16000)
    num_frames = math.ceil(len(audio) / 16000 * 25)

    setup_kwargs = {"smo_k_d": smo_k_d, "smo_k_s": smo_k_s}
    if enable_audio_reactive_sway:
        setup_kwargs["ctrl_info"] = _build_audio_reactive_ctrl_info(str(audio_path), num_frames)

    sdk = StreamSDK(DITTO_CFG_PKL, DITTO_DATA_ROOT)
    sdk.setup(str(reference_image_path), str(output_video_path), **setup_kwargs)

    _validate_ditto_internals(sdk)  # crash loud here, not silently-wrong later

    if expression_alpha > 0:
        _apply_expression_unlock(sdk, expression_alpha)
    if blink_scale != 1.0:
        _apply_blink_softening(sdk, blink_scale)

    sdk.setup_Nd(
        N_d=num_frames,
        fade_in=fade_in_frames if fade_in_frames > 0 else -1,
        fade_out=-1,
        ctrl_info=setup_kwargs.get("ctrl_info", {}),
    )
    aud_feat = sdk.wav2feat.wav2feat(audio)
    sdk.audio2motion_queue.put(aud_feat)
    sdk.close()

    cmd = (
        f'ffmpeg -loglevel error -y -i "{sdk.tmp_output_path}" '
        f'-i "{audio_path}" -map 0:v -map 1:a -c:v copy -c:a aac "{output_video_path}"'
    )
    ret = os.system(cmd)
    if ret != 0 or not output_video_path.exists():
        raise RuntimeError(
            f"Ditto inference completed but final ffmpeg mux failed "
            f"(exit {ret}) or output missing at {output_video_path}."
        )

    return {
        "reference_image": str(reference_image_path),
        "audio_path": str(audio_path),
        "output_path": str(output_video_path),
        "model": "Ditto",
        "smo_k_d": smo_k_d,
        "smo_k_s": smo_k_s,
        "fade_in_frames": fade_in_frames,
        "blink_scale": blink_scale,
        "expression_alpha": expression_alpha,
        "audio_reactive_sway": enable_audio_reactive_sway,
    }


def run_ditto_subprocess(reference_image_path, audio_path, output_video_path, **kwargs):
    """Runs this same file as a subprocess under DITTO_CONDA_ENV via
    `conda run`, for callers running in a different environment.
    Prefer run_ditto() directly if already inside DITTO_CONDA_ENV."""
    reference_image_path = Path(reference_image_path).resolve()
    audio_path = Path(audio_path).resolve()
    output_video_path = Path(output_video_path).resolve()
    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "conda", "run", "-n", DITTO_CONDA_ENV, "--no-capture-output",
        "python", str(Path(__file__).resolve()),
        str(reference_image_path), str(audio_path), str(output_video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Ditto subprocess failed:\n{result.stdout[-2000:]}\n---STDERR---\n{result.stderr[-3000:]}"
        )
    if not output_video_path.exists():
        raise RuntimeError(f"Ditto subprocess exited cleanly but no video was found at {output_video_path}.")

    return {
        "reference_image": str(reference_image_path),
        "audio_path": str(audio_path),
        "output_path": str(output_video_path),
        "model": "Ditto",
    }


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python stage4c_ditto.py <reference_image> <audio_wav_path> <output_video_path>")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    audio_path = Path(sys.argv[2])
    out_path = Path(sys.argv[3])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    start = time.time()
    params = run_ditto(image_path, audio_path, out_path)
    elapsed = time.time() - start

    print(f"Ditto animation completed in {elapsed:.1f}s -> {out_path}")
    print("CHECKPOINT:")
    print("  - lip sync matches the audio, mouth shapes look natural")
    print("  - head shows subtle, speech-reactive sway (not frozen)")
    print("  - some eyebrow/cheek movement visible, no obvious warping")
    print("  - blinks look natural, not abrupt")
    print("  - opening ~0.5s doesn't show a stuck-open mouth")
