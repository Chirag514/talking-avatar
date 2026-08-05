"""
stage2_voice_gen.py

Stage 2 — Voice Generation (OmniVoice, k2-fsa / Xiaomi)
Zero-shot voice cloning: no training step, just a reference clip + text.

REPLACES Chatterbox Multilingual V3. Reason: OmniVoice's non-autoregressive
diffusion-LM architecture generates the full sequence against a fixed
text alignment in one pass, which structurally avoids the AR-style
word-skipping/drift that Chatterbox and VoxCPM2 both exhibited on your
Hindi/English scripts during testing. It also natively covers ~25+
Indian languages vs. Chatterbox's Hindi-only.

CONFIRMED API (verified against github.com/k2-fsa/OmniVoice README,
huggingface.co/k2-fsa/OmniVoice model card, and your own tested notebook
cells, July 2026):
  - Single class, no separate "multilingual" variant needed — language
    is auto-detected from the input text itself:
        from omnivoice import OmniVoice
        model = OmniVoice.from_pretrained("k2-fsa/OmniVoice",
                                           device_map="cuda:0",
                                           dtype=torch.float16)
  - Generation: model.generate(text=..., ref_audio=..., ref_text=...,
                                speed=..., duration=..., language_id=...)
  - ref_text is optional — Whisper auto-transcribes ref_audio if omitted.
    We still pass it through when the caller supplies one, since exact
    transcripts give the most faithful clone.
  - duration (seconds) FORCES exact output length and, per official
    docs, OVERRIDES speed if both are set — do not pass both unless
    you intend duration to win.
  - Output: audio is a list of np.ndarray, shape (T,), at 24kHz.
    Confirmed 24kHz in your own test notebook (OmniVoice_TTs.ipynb).

DEPENDENCY NOTE (IMPORTANT — different from the old Chatterbox setup):
  OmniVoice needs transformers>=5.3.0 for voice-cloning mode. Your
  LatentSync stage only requires transformers>=4.51.0 (a floor, not an
  exact pin), so unlike Chatterbox — which hard-pinned
  transformers==5.2.0 and forced a version swap-out/swap-back dance —
  OmniVoice's floor requirement should coexist with LatentSync's floor
  requirement without needing that swap cycle. VERIFY this holds on
  your specific pod image before assuming it; see setup_runpod.sh for
  the install order this was tested against.

  The omnivoice PyPI package also declares torch==2.8.* as a
  dependency, which WILL downgrade a pinned torch install (e.g.
  2.6.0+cu124, if that's what your LatentSync setup needs) if installed
  normally. setup_runpod.sh installs it with --no-deps and adds the
  remaining requirements manually to protect whatever torch version
  your pod image is pinned to.

HF CACHE NOTE (RunPod-specific): by default, HuggingFace downloads
  OmniVoice's weights to ~/.cache/huggingface, which typically lives on
  local container disk — meaning a full multi-GB re-download every time
  the pod restarts. This module redirects HF_HOME to the persistent
  network volume (/workspace) if it isn't already set, so the download
  only happens once across the life of the volume, not once per pod
  session.

NO WATERMARK NOTE: unlike Chatterbox's PerTh watermark (embedded in
every output, cannot be disabled), OmniVoice's official docs do not
document any built-in watermarking. If provenance/detection matters
for your use case, that's now your responsibility to add downstream.

CLI NOTE (added — this used to be positional-args-only with no way to
set speed/duration/language without editing this file, which is a
problem now that you're testing standalone via SSH instead of typing
speed values into notebook cells by hand). Run:
    python stage2_voice_gen.py --help
for the full flag list.
"""

import inspect
import os
import time
from pathlib import Path

import soundfile as sf
import torch
import json as _json

# RunPod: redirect HuggingFace's cache to the persistent network volume
# so OmniVoice's multi-GB checkpoint survives pod restarts instead of
# re-downloading every session. Must be set BEFORE importing
# transformers/omnivoice, since HF reads this env var at import time.
# No-op if you've already set HF_HOME yourself (e.g. in setup_runpod.sh
# or your pod's env var template) — this only fills in a default.
os.environ.setdefault("HF_HOME", "/workspace/hf_cache")

from transformers.configuration_utils import PretrainedConfig

# ── Monkeypatch: transformers' PretrainedConfig.to_json_string() crashes
# on OmniVoice's checkpoint because its composite config embeds a raw
# Qwen3Config object as a value instead of a plain dict. transformers'
# own from_dict() calls logger.info(f"Model config {config}") during
# loading — that f-string evaluates config.__repr__() -> to_json_string()
# immediately regardless of log level, so it crashes even with logging
# silenced. This patch adds default=str as a json.dumps fallback so any
# non-serializable nested object gets stringified instead of crashing.
# This only affects the *logging representation* of the config, not the
# actual model weights/behavior — safe to apply.
def _patched_to_json_string(self, use_diff: bool = True) -> str:
    config_dict = self.to_diff_dict() if use_diff else self.to_dict()
    return _json.dumps(config_dict, indent=2, sort_keys=True, default=str) + "\n"

PretrainedConfig.to_json_string = _patched_to_json_string

from omnivoice import OmniVoice


# Loading the model is expensive — load once, reuse across calls.
# In the full pipeline this should be loaded once at orchestrator startup,
# not inside this function, to avoid reloading per video.
_model = None
_SAMPLE_RATE = 24000  # OmniVoice native output rate, confirmed


# Params generate_voice() actually passes to model.generate() -- keep this
# in sync with gen_kwargs below. Used only to VALIDATE, never to change
# what gets passed.
_EXPECTED_GENERATE_PARAMS = {"text", "ref_audio", "ref_text", "speed", "duration", "language_id"}


def _validate_generate_signature(model):
    """
    Fails loudly if the installed omnivoice version's model.generate()
    no longer accepts the parameter names this file was written
    against (confirmed July 2026 -- see module docstring).

    Why this matters more than a normal TypeError would catch: if
    generate() has a **kwargs catch-all, passing a renamed/removed
    parameter (e.g. ref_audio -> reference_audio) does NOT raise --
    it's silently swallowed, and OmniVoice quietly falls back to
    whatever its internal default is instead of cloning the reference
    voice. You'd get a video with a generic/wrong voice and no error
    anywhere. This check catches that case specifically.
    """
    sig = inspect.signature(model.generate)
    accepted = set(sig.parameters.keys())
    has_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    unconfirmed = _EXPECTED_GENERATE_PARAMS - accepted

    if not unconfirmed:
        return  # every param this file passes is in the explicit signature — good

    if has_var_kwargs:
        # Can't prove these are dropped vs. handled via **kwargs -- but can't
        # prove they're NOT silently dropped either. Warn loudly either way.
        print(
            f"WARNING: omnivoice's model.generate() accepts **kwargs, and "
            f"parameter(s) {sorted(unconfirmed)} are not in its explicit "
            f"signature {sorted(accepted)}. They may be silently ignored "
            f"rather than erroring, which would mean voice cloning/speed/"
            f"duration control silently doesn't work as expected. Manually "
            f"confirm against your installed omnivoice version before "
            f"trusting output — e.g. run a test with an obviously-wrong "
            f"--speed value and confirm the output pacing actually changes."
        )
    else:
        raise RuntimeError(
            f"Installed omnivoice's model.generate() does not accept "
            f"parameter(s) {sorted(unconfirmed)}, which generate_voice() in "
            f"this file relies on. Its current signature is: {sig}. The "
            f"omnivoice API has changed since this file was written "
            f"(confirmed against July 2026 docs/model card — see module "
            f"docstring). Update gen_kwargs in generate_voice() to match "
            f"the current signature before proceeding — otherwise this "
            f"would either crash on every call or, worse, silently drop "
            f"reference-voice cloning."
        )


def _get_model(device: str = "cuda:0"):
    global _model
    if _model is None:
        _model = OmniVoice.from_pretrained(
            "k2-fsa/OmniVoice", device_map=device, dtype=torch.float16,
            load_asr=True,  # pre-load Whisper ASR now, not lazily on first
                             # ref_text-omitted call — lazy-loading it inside
                             # generate() was the likely trigger for
                             # "TypeError: Object of type Qwen3Config is not
                             # JSON serializable"
        )
        _validate_generate_signature(_model)  # crash/warn loud here, not silently-wrong later
    return _model


def generate_voice(script_text: str, reference_voice_clip_path: str,
                    output_wav_path: Path, language: str = None,
                    reference_transcript: str = None,
                    speed: float = None, duration: float = None) -> dict:
    """
    Generates cloned speech audio from script_text in the voice of the
    speaker in reference_voice_clip_path.

    language: OPTIONAL. Only pass this if you've confirmed language_id
    works on your installed omnivoice version — passing it to
    model.generate() was found to trigger
    "TypeError: Object of type Qwen3Config is not JSON serializable"
    deep inside generate() on at least one tested setup. The official
    README's own Python API examples never pass language_id either
    (it only appears in the separate JSONL batch-inference format) —
    OmniVoice auto-detects language from script_text directly, so
    leaving this as None is the safe default.

    reference_transcript: exact transcript of what's said in
    reference_voice_clip_path. Optional — if omitted, OmniVoice's
    built-in Whisper ASR auto-transcribes the reference clip for you
    (a manual step your old Chatterbox setup required every time).

    speed: float multiplier on speaking rate (1.0 = default, 1.2 = ~20%
    faster). Ignored if duration is also set. This is the parameter
    your test notebook was setting by hand per-run (e.g. "speed=1.4" in
    output filenames) — it's now a proper CLI flag too, see __main__
    below / `python stage2_voice_gen.py --help`.

    duration: float, forces the output to exactly this many seconds.
    Overrides speed if both are provided.
    """
    model = _get_model()

    gen_kwargs = dict(
        text=script_text,
        ref_audio=reference_voice_clip_path,
    )
    if language:
        gen_kwargs["language_id"] = language
    if reference_transcript:
        gen_kwargs["ref_text"] = reference_transcript
    if duration is not None:
        gen_kwargs["duration"] = duration
    elif speed is not None:
        gen_kwargs["speed"] = speed

    audio = model.generate(**gen_kwargs)

    output_wav_path = Path(output_wav_path)
    output_wav_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_wav_path), audio[0], _SAMPLE_RATE)

    return {
        "language": language,
        "model_variant": "omnivoice",
        "sample_rate": _SAMPLE_RATE,
        "reference_clip": reference_voice_clip_path,
        "auto_transcribed_reference": reference_transcript is None,
        "speed": speed,
        "duration": duration,
        "output_path": str(output_wav_path),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Stage 2 standalone test — OmniVoice voice cloning. "
                    "Replaces the old notebook cells that set speed/duration "
                    "by hand per run; all of that is now a CLI flag.",
    )
    parser.add_argument("script_text", help="Path to a .txt file containing the script.")
    parser.add_argument("reference_voice", help="Path to the reference voice clip (wav/m4a/mp3).")
    parser.add_argument("--reference_transcript", default=None,
                         help="Optional path to a .txt file with the exact transcript of "
                              "reference_voice. If omitted, OmniVoice's built-in Whisper "
                              "ASR auto-transcribes it for you.")
    parser.add_argument("--speed", type=float, default=None,
                         help="Speaking-rate multiplier, e.g. 1.2 = ~20%% faster. "
                              "Ignored if --duration is also set. This is the parameter "
                              "the old notebook set by editing a variable per run — now "
                              "just pass e.g. --speed 1.4")
    parser.add_argument("--duration", type=float, default=None,
                         help="Force exact output length in seconds. Overrides --speed if both are set.")
    parser.add_argument("--language", default=None,
                         help="Usually leave unset — see the language= docstring above for "
                              "why (a known crash on at least one tested omnivoice version). "
                              "OmniVoice auto-detects language from the script text.")
    parser.add_argument("--output", default=None,
                         help="Where to write the output .wav. Defaults to "
                              "test_output/02_voice_speed<speed>_dur<duration>.wav so multiple "
                              "test runs with different speed/duration don't overwrite each other "
                              "(this replaces the old notebook habit of hand-naming output files "
                              "like 'OutputVoice_speed=1.4.wav').")
    args = parser.parse_args()

    script_text_content = Path(args.script_text).read_text()
    ref_transcript_content = (
        Path(args.reference_transcript).read_text() if args.reference_transcript else None
    )

    if args.output:
        out_path = Path(args.output)
    else:
        tag = f"_speed{args.speed}" if args.speed is not None else ""
        tag += f"_dur{args.duration}" if args.duration is not None else ""
        out_path = Path(f"test_output/02_voice{tag or '_default'}.wav")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    start = time.time()
    params = generate_voice(
        script_text_content,
        args.reference_voice,
        out_path,
        language=args.language,
        reference_transcript=ref_transcript_content,
        speed=args.speed,
        duration=args.duration,
    )
    elapsed = time.time() - start

    print(f"Generated voice clone in {elapsed:.1f}s -> {out_path}")
    print(f"Speed: {args.speed}  Duration: {args.duration}  Language: {args.language}")
    print(f"Reference auto-transcribed by Whisper: {params['auto_transcribed_reference']}")
    print("CHECKPOINT: listen to the output file. Check for:")
    print("  - speaker similarity to the reference clip")
    print("  - hallucinated/off-prompt words")
    print("  - unnatural pacing or robotic artifacts (try a different --speed if pacing is off)")


# HF_HOME=~/.cache/huggingface python stage2_voice_gen.py /path/to/path_a_pipeline/text_script_yt_video_2.txt
# /path/to/path_a_pipeline/LingaSir_ReferenceVoice.wav --speed 1.4 --output /path/to/path_a_pipeline/Output_Ling
# aSir_YT_2_speed=1.4.wav