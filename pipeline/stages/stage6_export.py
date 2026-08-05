"""
stage6_export.py

Stage 6 — Final Export (FFmpeg)
Muxes the chosen video (restored or unrestored, from Stage 5's A/B
output) with the cloned audio into the final deliverable.

Note: Stage 4c (Ditto) already muxes the cloned audio into its raw
output video, and Stage 5 (restoration) carries that same audio
through when it re-encodes frames. So by the time this stage runs,
video_path already contains the correct audio track, and audio_path
is the same underlying clip. This re-mux is therefore redundant in
the common case (it just re-encodes audio that's already correct) --
it's kept as a safety net in case restoration is skipped or audio
ever gets dropped upstream, not because it's doing essential work.
If you want to avoid the extra re-encode, verify video_path already
has the right audio track and skip this stage's mux in that case.
"""

import subprocess
from pathlib import Path


def export_final(video_path: Path, audio_path: Path, output_path: Path) -> dict:
    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        "-y",
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg export failed:\n{result.stderr}")

    if not output_path.exists():
        raise RuntimeError("FFmpeg exited cleanly but no output file was found.")

    return {"video_source": str(video_path), "audio_source": str(audio_path)}


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python stage6_export.py <video_path> <audio_path>")
        sys.exit(1)

    video_path = Path(sys.argv[1])
    audio_path = Path(sys.argv[2])
    out_path = Path("test_output/06_final.mp4")

    export_final(video_path, audio_path, out_path)
    print(f"Final video exported -> {out_path}")
    print("CHECKPOINT: play the full final video, confirm audio/video sync")
    print("holds from start to end, not just the first few seconds.")
