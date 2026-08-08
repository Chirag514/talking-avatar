"""
stage1_safety_gate.py

Stage 1 — Safety Gate
Two independent checks, both must pass before generation proceeds:

  1. Content moderation (OpenAI Moderation API) — catches policy-violating
     SCRIPT TEXT. This does not know or care who the voice/face belongs to.
  2. Consent verification — catches NON-CONSENSUAL LIKENESS USE. A perfectly
     polite script can still be an unauthorized deepfake of someone who
     never agreed to it.

IMPORTANT — VERIFY ON YOUR BOX:
- `pip install openai` and confirm the current client syntax. The library's
  API has changed across major versions (0.x vs 1.x client style) — check
  the installed version's own README/quickstart rather than assuming the
  call shape below is current.
- The model name 'omni-moderation-latest' should be confirmed against
  OpenAI's current moderation docs before relying on it in production —
  model names get superseded.

CONSENT VERIFICATION — explicit attestation, not identity verification:
check_consent() requires an explicit consent_confirmed=True (wired to
--i_confirm_consent on the CLI) and logs who attested, to which files,
and when. It does NOT verify the attestation is true — it cannot
confirm the person in reference_face_image_path is who
submitting_user_id claims, or that they actually consented. Before any
public launch, upgrade this to a real self-upload match + liveness
check. What this DOES fix vs. a bare stub: no attestation means no
run — it's no longer silently assumed True with zero record of it.
"""

from openai import OpenAI  # VERIFY this import path against your installed openai version

# On a rented GPU pod, set OPENAI_API_KEY as an environment variable
# before running the pipeline — either via the provider's pod template
# env var settings, or with `export OPENAI_API_KEY=...` in your shell/
# setup script before invoking run_pipeline.py. This module just reads
# it from the environment; it doesn't manage secrets itself.
client = OpenAI()  # picks up OPENAI_API_KEY from environment


def check_moderation(script_text: str) -> dict:
    """
    Returns {"passed": bool, "flagged_categories": [...], "raw": <api response dict>}
    """
    response = client.moderations.create(
        model="omni-moderation-latest",  # VERIFY current model name
        input=script_text,
    )
    result = response.results[0]

    flagged_categories = [
        category for category, flagged in result.categories.__dict__.items()
        if flagged
    ]

    return {
        "passed": not result.flagged,
        "flagged_categories": flagged_categories,
    }


def check_consent(submitting_user_id: str, reference_voice_clip_path: str,
                   reference_face_image_path: str,
                   consent_confirmed: bool = False) -> dict:
    """
    Real explicit-attestation check — NOT identity/liveness verification.

    This requires the caller to have explicitly passed consent_confirmed=True
    (wired to --i_confirm_consent on the CLI, see run_pipeline.py). It logs
    who attested, to what files, and when. What it does NOT do: verify the
    attestation is true. It cannot confirm the person in reference_face_image_path
    is who submitting_user_id claims, or that they actually agreed to this use.
    Before any public launch, upgrade this to an actual self-upload match
    (compare submitted reference clip/photo against a verified account
    photo) and ideally a liveness check -- track that as a blocking task,
    not a v1 nice-to-have. This function only ensures a real "yes, I attest"
    signal is required and recorded, instead of being silently assumed.
    """
    import datetime
    import hashlib
    from pathlib import Path

    if not consent_confirmed:
        return {
            "passed": False,
            "method": "explicit_attestation_required",
            "reason": "No consent attestation provided -- pass --i_confirm_consent "
                      "(CLI) or consent_confirmed=True (API) to proceed. This is "
                      "not optional: generating a cloned voice/likeness without "
                      "an explicit attestation is not supported by this pipeline.",
        }

    def _file_hash(path):
        try:
            return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
        except Exception:
            return "unavailable"

    record = {
        "passed": True,
        "method": "explicit_attestation_only_NOT_IDENTITY_VERIFIED",
        "submitting_user_id": submitting_user_id,
        "attested_at": datetime.datetime.utcnow().isoformat() + "Z",
        "reference_voice_clip_hash": _file_hash(reference_voice_clip_path),
        "reference_face_image_hash": _file_hash(reference_face_image_path),
        "warning": "This confirms a consent CLAIM was explicitly made and logged. "
                   "It does NOT verify the claim is true -- no identity or "
                   "liveness check has been performed.",
    }
    print(f"[stage1_safety_gate] Consent attestation recorded: user={submitting_user_id}, "
          f"voice_hash={record['reference_voice_clip_hash']}, "
          f"face_hash={record['reference_face_image_hash']}, "
          f"at={record['attested_at']}")
    return record


def run_safety_gate(script_text: str, submitting_user_id: str,
                     reference_voice_clip_path: str,
                     reference_face_image_path: str,
                     consent_confirmed: bool = False) -> dict:
    """
    Runs both checks. Pipeline should abort if either fails.
    """
    moderation_result = check_moderation(script_text)
    consent_result = check_consent(
        submitting_user_id, reference_voice_clip_path, reference_face_image_path,
        consent_confirmed=consent_confirmed,
    )

    overall_pass = moderation_result["passed"] and consent_result["passed"]

    return {
        "passed": overall_pass,
        "moderation": moderation_result,
        "consent": consent_result,
    }


if __name__ == "__main__":
    # Standalone test (Step 1 of the plan). Run a few scripts through this
    # alone, including at least one you expect to fail, before wiring it
    # into the full pipeline.
    test_scripts = [
        "Hi, welcome to our product demo. Today I'll walk you through three features.",
        # Add a borderline/policy-violating test case here yourself —
        # don't paste real harmful content into a test file.
    ]

    for script in test_scripts:
        result = check_moderation(script)
        print(f"Script: {script[:50]}...")
        print(f"  Passed: {result['passed']}")
        if result["flagged_categories"]:
            print(f"  Flagged categories: {result['flagged_categories']}")
        print()

    print("CHECKPOINT: confirm pass/fail behavior matches expectations on your "
          "own borderline test cases before moving on.")
