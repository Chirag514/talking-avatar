"""
stage5_restoration.py

Stage 5 — Restoration & Upscale (Real-ESRGAN, optional)
Per the architecture doc: run WITH and WITHOUT this stage, compare,
and only make it default if the quality gain is clearly visible.
That's why run_restoration() here always produces both outputs rather
than picking one — the A/B comparison is the point, not a side effect.

DO NOT enable --face_enhance (GFPGAN). The doc flags GFPGAN's DFDNet
dependency (CC BY-NC-SA 4.0) as an unresolved commercial-use question.
This script deliberately never passes that flag.

IMPORTANT — VERIFY ON YOUR BOX:
Real-ESRGAN is typically run via its `inference_realesrgan.py` script
or a similarly named CLI entry point — confirm the exact script name,
the model weights filename (the doc references the "x4plus" model),
and current argument names against the xinntao/Real-ESRGAN repo.
"""

import shutil
import subprocess
import time
from pathlib import Path

# RunPod: model weights/repo live on the persistent network volume
# (/workspace) so they survive pod restarts without re-cloning/re-
# downloading. Copied once per session to fast local container disk
# for actual inference, since /workspace is network-attached storage
# and slower than local NVMe for the many small reads inference does.
PERSISTENT_REALESRGAN_REPO_PATH = "/workspace/models/Real-ESRGAN"  # adjust if you cloned elsewhere
LOCAL_REALESRGAN_REPO_PATH = "/root/Real-ESRGAN"  # fast local scratch cache, see _get_local_repo_path()
MODEL_NAME = "RealESRGAN_x4plus"  # VERIFY against repo's available model weights

_local_repo_ready = False  # module-level cache flag, mirrors stage2's _get_model() pattern


def _get_local_repo_path() -> str:
    """
    Copies the Real-ESRGAN repo (code + model weights) from the
    persistent network volume to fast local container disk ONCE per
    session, then reuses the local copy for every subsequent call.
    Model weight files are read on every inference_realesrgan.py
    invocation; leaving them on the network volume means paying its
    higher latency on every single pipeline run instead of once. Falls
    back to the persistent-volume path directly if the local copy
    can't be created for some reason (e.g. insufficient local disk
    space), so this degrades gracefully rather than failing the whole
    stage.
    """
    global _local_repo_ready
    local_path = Path(LOCAL_REALESRGAN_REPO_PATH)

    if _local_repo_ready and local_path.exists():
        return str(local_path)

    try:
        if not local_path.exists():
            shutil.copytree(PERSISTENT_REALESRGAN_REPO_PATH, local_path)
        _local_repo_ready = True
        return str(local_path)
    except Exception as e:
        print(f"WARNING: could not stage Real-ESRGAN locally ({e}); "
              f"falling back to reading directly from the network volume (slower).")
        return PERSISTENT_REALESRGAN_REPO_PATH


# def run_restoration(input_video_path: Path, output_video_path: Path,
#                      scale: int = 2) -> dict:
#     """
#     Runs Real-ESRGAN on input_video_path. NEVER passes --face_enhance.
#     """
#     cmd = [
#         "python", f"{REALESRGAN_REPO_PATH}/inference_realesrgan.py",  # VERIFY script name
#         "-i", str(input_video_path),                                   # VERIFY arg name
#         "-o", str(output_video_path.parent),                           # VERIFY arg name
#         "-n", MODEL_NAME,                                              # VERIFY arg name
#         "-s", str(scale),                                              # VERIFY arg name
#         # deliberately no --face_enhance — see GFPGAN warning above
#     ]

#     result = subprocess.run(cmd, capture_output=True, text=True, cwd=REALESRGAN_REPO_PATH)
#     if result.returncode != 0:
#         raise RuntimeError(f"Real-ESRGAN failed:\n{result.stderr}")

#     return {"scale": scale, "model": MODEL_NAME, "face_enhance": False}

def run_restoration(input_video_path: Path, output_video_path: Path, scale: int = 2) -> dict:
    import tempfile, shutil, subprocess

    # 1. Extract frames from input video
    frames_in  = Path(tempfile.mkdtemp())
    frames_out = Path(tempfile.mkdtemp())

    subprocess.run([
        "ffmpeg", "-y", "-i", str(input_video_path),
        str(frames_in / "%05d.png")
    ], check=True, capture_output=True)

    # 2. Run Real-ESRGAN on the frames directory
    cmd = [
        "python", "inference_realesrgan.py",
        "-i", str(frames_in),
        "-o", str(frames_out),
        "-s", str(scale),
        "-n", MODEL_NAME,
    ]
    repo_path = _get_local_repo_path()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=repo_path)
    if result.returncode != 0:
        raise RuntimeError(f"Real-ESRGAN failed:\n{result.stderr}")

    # 2b. VERIFY output frames actually match what step 4 below assumes,
    # before muxing. Real-ESRGAN's default naming is "{name}_out.{ext}"
    # (confirmed against xinntao/Real-ESRGAN at time of writing) -- if
    # that convention or the frame count silently changed, muxing an
    # incomplete/mismatched set with ffmpeg's pattern glob wouldn't
    # necessarily error; it could produce a shorter/corrupted video
    # that LOOKS fine at a glance. Fail loud here instead.
    input_frame_count = len(list(frames_in.glob("*.png")))
    output_files = sorted(frames_out.glob("*.png"))
    if len(output_files) != input_frame_count:
        raise RuntimeError(
            f"Real-ESRGAN produced {len(output_files)} output frame(s) but "
            f"{input_frame_count} were extracted from the source video. "
            f"Either some frames failed silently, or Real-ESRGAN's output "
            f"naming convention no longer matches what this code assumes. "
            f"Inspect {frames_out} directly before proceeding -- muxing "
            f"a mismatched frame set would produce a corrupted or "
            f"truncated video without any ffmpeg error."
        )

    expected_first = frames_out / "00001_out.png"
    if not expected_first.exists():
        sample = [f.name for f in output_files[:5]]
        raise RuntimeError(
            f"Expected Real-ESRGAN's first output frame at "
            f"{expected_first} (naming pattern '{{name}}_out.png', which "
            f"the ffmpeg re-mux command below is hardcoded to expect), "
            f"but it doesn't exist. Actual filenames present: {sample}"
            f"{'...' if len(output_files) > 5 else ''}. Real-ESRGAN's "
            f"naming convention has changed -- update the '-i' pattern "
            f"in the ffmpeg command below (step 4) to match before "
            f"trusting this stage's output."
        )

    # 3. Get fps from original video
    probe = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(input_video_path)
    ], capture_output=True, text=True)
    fps_str = probe.stdout.strip()  # e.g. "25/1"
    num, den = fps_str.split("/")
    fps = float(num) / float(den)

    # 4. Re-mux upscaled frames + original audio back to video
    subprocess.run([
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(frames_out / "%05d_out.png"),
        "-i", str(input_video_path),
        "-map", "0:v", "-map", "1:a",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        str(output_video_path)
    ], check=True, capture_output=True)

    shutil.rmtree(frames_in)
    shutil.rmtree(frames_out)

    return {"scale": scale, "model": MODEL_NAME, "face_enhance": False}


def run_restoration_ab(lipsync_output_path: Path, run_dir: Path, scale: int = 2) -> dict:
    """
    Produces BOTH the restored and unrestored versions side by side,
    per the doc's A/B-before-committing recommendation.

    Returns paths to both for manual comparison.
    """
    unrestored_path = run_dir / "05_unrestored.mp4"
    restored_path = run_dir / "05_restored.mp4"

    # The "unrestored" version is just a copy of Stage 4A's output, kept
    # under a Stage-5-named file so both live side by side in the run folder.
    shutil.copy(lipsync_output_path, unrestored_path)

    params = run_restoration(lipsync_output_path, restored_path, scale=scale)

    return {
        "unrestored_path": str(unrestored_path),
        "restored_path": str(restored_path),
        "params": params,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python stage5_restoration.py <lipsync_output_video>")
        sys.exit(1)

    lipsync_output = Path(sys.argv[1])
    run_dir = Path("test_output")
    run_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    result = run_restoration_ab(lipsync_output, run_dir)
    elapsed = time.time() - start

    print(f"Restoration A/B completed in {elapsed:.1f}s")
    print(f"  Unrestored: {result['unrestored_path']}")
    print(f"  Restored:   {result['restored_path']}")
    print("CHECKPOINT: compare both side by side. Only adopt Real-ESRGAN as")
    print("default if the quality gain is clearly visible, per the doc's guidance.")
