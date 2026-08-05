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

CONSENT VERIFICATION — NOT IMPLEMENTED HERE:
The function below is a stub. At minimum it should require the submitting
user to confirm the face/voice is their own (self-attestation + upload
matching), and the doc's own recommendation is to add liveness or
identity-match checks before any public launch. Do not ship this stub
to production — it always returns True, which means "no consent check
actually happened."
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
                   reference_face_image_path: str) -> dict:
    """
    STUB — replace before any public launch.

    Today this only checks that the user explicitly clicked "this is my
    voice and face" — it does NOT verify that claim. Real implementation
    needs at minimum a self-upload match (compare submitted reference
    clip/photo against a verified account photo) and ideally a liveness
    check. Track this as a blocking task before launch, not a v1 nice-to-have.
    """
    # TODO: real identity/liveness check goes here
    self_attested = True  # placeholder — assumes the UI already collected this

    return {
        "passed": self_attested,
        "method": "self_attestation_only_NOT_VERIFIED",
    }


def run_safety_gate(script_text: str, submitting_user_id: str,
                     reference_voice_clip_path: str,
                     reference_face_image_path: str) -> dict:
    """
    Runs both checks. Pipeline should abort if either fails.
    """
    moderation_result = check_moderation(script_text)
    consent_result = check_consent(
        submitting_user_id, reference_voice_clip_path, reference_face_image_path
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
