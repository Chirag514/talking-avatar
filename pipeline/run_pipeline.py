"""
run_pipeline.py

THE single entrypoint for Path A: reference image + text script + a
reference voice clip -> a finished talking-avatar video.

This replaced two earlier, separate entrypoints:
  - the old run_pipeline.py (Path A / template-video / LatentSync route)
  - run_pipeline_image.py (Path B / image / EchoMimicV3 route)
Both source stages (stage4a_lipsync.py, stage4b_echomimic.py) and the
router that chose between them (stage3_router.py) have been retired.
Ditto (stage4c_ditto.py) is the only animation stage now -- see the
Trials document for why it was selected over every alternative tried.

Stages, in order:
  1. Safety gate      (stage1_safety_gate.py)   -- moderation + consent
  2. Voice generation  (stage2_voice_gen.py)      -- OmniVoice cloning
  3. Animation         (stage4c_ditto.py)         -- Ditto, image+audio -> video
  4. Restoration       (stage5_restoration.py)    -- optional, Real-ESRGAN upscale
  5. Overlay compositing (stage7_overlays.py)     -- optional, animated callouts
  6. Export            (stage6_export.py)         -- final mux/output

IMPORTANT -- environments: stage2 (OmniVoice) and stage4c (Ditto) each
need their own conda environment (see this repo's top-level README for
exact setup). This script assumes it is being run from an environment
that can import stage1/stage2/stage5/stage6/stage7 directly, and calls
into Ditto via run_ditto_subprocess() (which shells out to the `ditto`
conda env) rather than importing it directly -- so you do NOT need to
run this top-level script itself from inside the `ditto` env. If you'd
rather import run_ditto() directly and skip the subprocess hop, run
this whole script from inside the `ditto` env instead and swap the
call below.
"""

import argparse
import sys
import time
from pathlib import Path

from stages.stage1_safety_gate import run_safety_gate
from stages.stage2_voice_gen import generate_voice
from stages.stage4c_ditto import run_ditto_subprocess
from stages.stage5_restoration import run_restoration
from stages.stage6_export import export_final
from stages.stage7_overlays import run_overlays


def run_pipeline(script_text: str, reference_image_path: Path,
                  reference_voice_clip_path: Path, output_path: Path,
                  submitting_user_id: str = "unknown",
                  speed: float = None,
                  enable_restoration: bool = True,
                  enable_overlays: bool = False,
                  enable_safety_gate: bool = True,
                  consent_confirmed: bool = False,
                  work_dir: Path = None) -> dict:
    """
    Runs the full pipeline end to end. Raises on any stage failure or
    safety-gate rejection -- callers should catch and surface the
    specific stage/reason, not swallow failures silently.

    consent_confirmed=True is REQUIRED (when enable_safety_gate=True) --
    wired to --i_confirm_consent on the CLI. This is an explicit
    attestation, not identity verification: it does not confirm the
    person in reference_image_path/reference_voice_clip_path actually
    is who submitting_user_id claims or that they consented, only that
    someone explicitly claimed so and it was logged (see
    stage1_safety_gate.py's check_consent()). Still real -- a hardcoded
    True with no record was the previous state; requiring and logging
    an explicit flag is a meaningful floor, not a formality.

    enable_safety_gate=False skips Stage 1 entirely, including the
    consent attestation requirement (no OPENAI_API_KEY needed either).
    Dev/testing convenience only -- do not leave this off for anything
    beyond your own local testing, and never for a real person's
    likeness/voice.
    """
    work_dir = Path(work_dir) if work_dir else output_path.parent / f"run_{int(time.time())}"
    work_dir.mkdir(parents=True, exist_ok=True)

    gate_result = None
    if enable_safety_gate:
        print("[1/6] Safety gate...")
        gate_result = run_safety_gate(
            script_text=script_text,
            submitting_user_id=submitting_user_id,
            reference_voice_clip_path=str(reference_voice_clip_path),
            reference_face_image_path=str(reference_image_path),
            consent_confirmed=consent_confirmed,
        )
        if not gate_result["passed"]:
            raise RuntimeError(f"Safety gate rejected this request: {gate_result}")
    else:
        print("[1/6] WARNING: Safety gate SKIPPED (enable_safety_gate=False) -- "
              "script text was NOT moderated, and NO consent attestation was "
              "recorded. Dev/testing only.")

    print("[2/6] Voice generation (OmniVoice)...")
    cloned_audio_path = work_dir / "cloned_voice.wav"
    voice_result = generate_voice(
        script_text=script_text,
        reference_voice_clip_path=str(reference_voice_clip_path),
        output_wav_path=cloned_audio_path,
        speed=speed,
    )

    print("[3/6] Animation (Ditto)...")
    raw_video_path = work_dir / "raw_animation.mp4"
    ditto_result = run_ditto_subprocess(
        reference_image_path=reference_image_path,
        audio_path=cloned_audio_path,
        output_video_path=raw_video_path,
    )

    current_video_path = raw_video_path
    restoration_result = None
    if enable_restoration:
        print("[4/6] Restoration (Real-ESRGAN)...")
        restored_video_path = work_dir / "restored.mp4"
        restoration_result = run_restoration(current_video_path, restored_video_path)
        current_video_path = restored_video_path
    else:
        print("[4/6] Restoration skipped (enable_restoration=False)")

    overlays_result = None
    if enable_overlays:
        print("[5/6] Overlay compositing...")
        overlaid_video_path = work_dir / "with_overlays.mp4"
        overlays_result = run_overlays(
            video_path=current_video_path,
            audio_path=cloned_audio_path,
            script_text=script_text,
            output_path=overlaid_video_path,
        )
        current_video_path = overlaid_video_path
    else:
        print("[5/6] Overlays skipped (enable_overlays=False)")

    print("[6/6] Export...")
    export_result = export_final(
        video_path=current_video_path,
        audio_path=cloned_audio_path,
        output_path=output_path,
    )

    return {
        "output_path": str(output_path),
        "safety_gate": gate_result,
        "voice_generation": voice_result,
        "animation": ditto_result,
        "restoration": restoration_result,
        "overlays": overlays_result,
        "export": export_result,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Path A avatar pipeline: image + script + voice -> video")
    parser.add_argument("script_text_file", type=Path, help="path to a .txt file containing the script")
    parser.add_argument("reference_image", type=Path, help="reference photo to animate")
    parser.add_argument("reference_voice_clip", type=Path, help="reference voice clip to clone")
    parser.add_argument("output_path", type=Path, help="where the final video should be saved")
    parser.add_argument("--submitting_user_id", type=str, default="unknown")
    parser.add_argument("--speed", type=float, default=None)
    parser.add_argument("--no_restoration", action="store_true")
    parser.add_argument("--overlays", action="store_true")
    parser.add_argument("--i_confirm_consent", action="store_true",
                         help="REQUIRED (unless --skip_safety_gate is also set): explicit "
                              "attestation that the person in reference_image/"
                              "reference_voice_clip has consented to this use. This is "
                              "logged (see stage1_safety_gate.py's check_consent()). It "
                              "is an attestation, not identity verification -- attesting "
                              "falsely does not become true because you passed this flag.")
    parser.add_argument("--skip_safety_gate", action="store_true",
                         help="Skip Stage 1 entirely -- no OPENAI_API_KEY needed and no "
                              "consent attestation required either. Dev/testing only, "
                              "never for a real person's likeness/voice.")
    args = parser.parse_args()

    if not args.skip_safety_gate and not args.i_confirm_consent:
        print("ERROR: --i_confirm_consent is required (attesting the reference "
              "image/voice's subject has consented to this use), unless you pass "
              "--skip_safety_gate for local dev/testing.")
        sys.exit(1)

    if not args.script_text_file.exists():
        print(f"Script file not found: {args.script_text_file}")
        sys.exit(1)
    script_text = args.script_text_file.read_text()

    start = time.time()
    result = run_pipeline(
        script_text=script_text,
        reference_image_path=args.reference_image,
        reference_voice_clip_path=args.reference_voice_clip,
        output_path=args.output_path,
        submitting_user_id=args.submitting_user_id,
        speed=args.speed,
        enable_restoration=not args.no_restoration,
        enable_overlays=args.overlays,
        enable_safety_gate=not args.skip_safety_gate,
        consent_confirmed=args.i_confirm_consent,
    )
    elapsed = time.time() - start

    print(f"\nPipeline completed in {elapsed:.1f}s -> {args.output_path}")
