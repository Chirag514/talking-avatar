"""
stage7_overlays.py

Stage 7 — Animated Overlay Compositing
Adds timed, animated graphics on top of the final lip-synced video —
the "HeyGen-style" explainer graphics you originally pointed at:
single label callouts, side-by-side comparisons ("AI Agent = Employee"
vs "Agentic AI = Team"), and connected flow diagrams ("Planning ->
Coordination -> Decision"). Runs AFTER Stage 6's final export.

REV 3 — adds the two diagram types that were actually missing. REV 2
only supported single independent labels; your feedback was that real
relationship diagrams (comparisons, connected flows) were the point,
not standalone labels. This stayed on the Pillow/MoviePy toolchain
rather than switching to browser-rendered graphics, because that path
needed a live browser preview loop to debug that wasn't available at
the time.

REV 4 — that constraint no longer applies (headless screenshot capture
works fine without a live preview loop, just a slower iterate-and-view
cycle), and the actual complaint was that REV 3's output didn't look
professional/HeyGen-style. Two real upgrades, both in new sibling
modules rather than crammed into this file:

  overlay_render.py — renders each card as real HTML/CSS (glassmorphism
  blur-behind, layered box-shadow, gradient icon tile, cubic-bezier
  entrance easing, per-word blur-stagger reveal) via headless Chromium
  (Playwright), captured as a PNG-per-frame sequence rather than one
  static image pushed through a linear affine transform. This is a
  real ceiling difference from Pillow: no backdrop-filter, no bezier
  easing, no layered shadows without hand-rolling each pixel op.
  Long labels now WRAP to two lines instead of shrinking font size
  indefinitely. composite_overlays() below turns each frame sequence
  into a MoviePy clip via build_moviepy_layer() (ImageSequenceClip +
  a matching alpha-mask ImageSequenceClip), replacing the old
  _animated_layer() affine slide/scale wrapper — entrance, hold, AND
  exit fade are now baked into the rendered frames themselves.

  overlay_placement.py — analyze_frame_for_overlay() only ever looked
  at the video frame at an event's START time. If the speaker moved
  during the 2-4s an overlay stays on screen, placement that was safe
  at t=start could end up overlapping the face by t=end. The new
  analyze_region_for_overlay() samples several frames across the
  event's full [start, end] window and uses the UNION of all detected
  face boxes, so placement stays clear for the entire time the overlay
  is visible — not just its first frame. It also scores every safe
  candidate region (above/left/right) by visual busyness (Laplacian
  edge density) and picks the quietest one, instead of always
  preferring "above" first regardless of what's actually back there.

  Both were prototyped and round-trip tested against a synthetic
  moving-subject clip before being wired in here — see the demo this
  was validated against. Still recommend using preview_placement_on_video()
  against your OWN real footage before trusting this on a full run,
  same as REV 3's own advice.

THREE EVENT TYPES NOW:
  "label"      — single callout (REV 2's original behavior)
  "comparison" — two related concepts side by side, e.g. "AI Agent =
                 Employee" vs "Agentic AI = Team"
  "flow"       — 2-4 connected concepts revealed in sequence with
                 connector arrows, e.g. "Planning -> Coordination ->
                 Decision"

The LLM only picks ONE anchor timestamp per comparison/flow group (not
one per node) — asking it for several precise sub-timestamps per
concept risked nonsense. Instead, code deterministically staggers each
node's reveal by a fixed delay from that one anchor point. This is a
reliability tradeoff: simpler and safer, at the cost of the LLM not
controlling exact per-node timing.

THREE-PART PIPELINE:
  1. Word-level timestamps from the final audio (Whisper) — unchanged, confirmed working
  2. LLM-driven event detection via Groq — extended schema, same strict
     relevance criteria as REV 2
  3. Compositing animated overlays onto the video (Pillow + MoviePy) —
     extended with comparison/flow renderers

HONESTY NOTE: the comparison/flow layouts below (box positions, stagger
timing, arrow style) are a first design pass, not tuned against your
actual video framing. Use the new preview_comparison_frame() and
preview_flow_frames() helpers to check these BEFORE a full pipeline
run — same fast screenshot-and-fix loop that worked for the font bug.

GROQ MODEL NOTE: "llama-3.3-70b-versatile" is my best current guess at
a Groq-hosted model capable of reliable structured JSON output, NOT
independently verified against Groq's live model list at the time you
run this. Check https://console.groq.com/docs/models yourself.

DEPENDENCIES:
  pip install openai-whisper moviepy pillow groq

REQUIRED ENV: GROQ_API_KEY (https://console.groq.com/keys)
"""

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import whisper
from groq import Groq
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import VideoFileClip, ImageClip, CompositeVideoClip

from .overlay_placement import analyze_region_for_overlay
from .overlay_render import (
    render_animated_label_frames,
    render_animated_comparison_frames,
    build_moviepy_layer,
)

client = Groq()  # picks up GROQ_API_KEY from environment

_whisper_model = None  # module-level cache, same pattern as stage2's _get_model()


def _get_whisper_model(size: str = "small"):
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = whisper.load_model(size)
    return _whisper_model


def extract_word_timings(audio_path: Path) -> list:
    """Transcribes audio_path with word-level timestamps. Confirmed working
    in your own testing — returns [{"word", "start", "end"}, ...]."""
    model = _get_whisper_model()
    result = model.transcribe(str(audio_path), word_timestamps=True)

    word_timings = []
    for segment in result["segments"]:
        for word_info in segment.get("words", []):
            word_timings.append({
                "word": word_info["word"].strip(),
                "start": float(word_info["start"]),
                "end": float(word_info["end"]),
            })

    if not word_timings:
        raise RuntimeError(
            "No word-level timings extracted — check that your installed "
            "whisper version actually supports word_timestamps=True."
        )
    return word_timings


def _resolve_events_from_indices(raw_events: list, word_timings: list) -> list:
    """
    Converts the model's index-based output (anchor_index only) into
    full event dicts with real "start"/"end" times looked up from
    word_timings — the model never has to produce a timestamp number
    itself either, which also removes any risk of hallucinated timing,
    not just the Unicode-escape crash this was built to fix.

    Malformed/out-of-range events are skipped with a warning rather
    than crashing the whole run — one bad event from the model
    shouldn't take down everything else it got right.
    """
    resolved = []
    for e in raw_events:
        try:
            etype = e.get("type", "label")
            idx = e["anchor_index"]
            if not (0 <= idx < len(word_timings)):
                print(f"WARNING: skipping event with out-of-range anchor_index {idx}: {e}")
                continue
            anchor_word = word_timings[idx]

            if etype == "label":
                resolved.append({
                    "type": "label",
                    "start": anchor_word["start"],
                    "end": anchor_word["start"] + 2.0,
                    "label_text": e["label_text"],
                    "icon": e.get("icon", "box"),
                })
            elif etype == "comparison":
                resolved.append({
                    "type": "comparison",
                    "start": anchor_word["start"],
                    "end": anchor_word["start"] + 4.0,
                    "left": e["left"],
                    "right": e["right"],
                })
            elif etype == "flow":
                nodes = e.get("nodes", [])
                num_nodes = max(len(nodes), e.get("num_nodes", len(nodes)))
                resolved.append({
                    "type": "flow",
                    "start": anchor_word["start"],
                    "end": anchor_word["start"] + 1.5 * max(num_nodes, 1),
                    "nodes": nodes,
                })
            else:
                print(f"WARNING: skipping event with unknown type '{etype}': {e}")
        except (KeyError, TypeError) as parse_err:
            print(f"WARNING: skipping malformed event ({parse_err}): {e}")
            continue

    return resolved


def detect_callout_events(script_text: str, word_timings: list,
                           max_events: int = 5, max_retries: int = 2) -> list:
    """
    REV 4 — fixes a real crash found in production: Groq's model
    occasionally generated a MALFORMED Unicode escape (e.g. "\\u092p" —
    not valid hex) while trying to echo Hindi/Devanagari text back
    inside a JSON string for "anchor_word", causing Groq's json_object
    validation to reject the whole response with a 400 error. This is a
    real, somewhat unpredictable model failure mode with non-Latin
    scripts in JSON string fields — not reliably fixable with a prompt
    tweak alone.

    FIX: the model no longer echoes any script text back at all. Since
    we already HAVE the exact word list in word_timings, the model just
    references words BY INDEX NUMBER — integers can't have malformed
    Unicode escapes. start/end times are computed in Python from the
    real word_timings data afterward, not trusted from the model's own
    output either, which also eliminates any risk of hallucinated
    timestamps as a side benefit.

    Also adds retry_count retries with a fresh request on ANY failure
    (malformed JSON, API error, etc.) before giving up and returning an
    empty list rather than crashing your whole pipeline run over one
    bad generation.
    """
    # Indexed timing list — the model refers to words by their index
    # in THIS list, never by reproducing the word's text itself.
    indexed_words = [
        {"i": idx, "word": w["word"], "start": round(w["start"], 2), "end": round(w["end"], 2)}
        for idx, w in enumerate(word_timings)
    ]

    prompt = f"""You are selecting moments in a spoken script for animated
on-screen graphics in an explainer video. Three types of graphic are
available — pick whichever fits each moment, don't force a type that
doesn't fit.

IMPORTANT: refer to words ONLY by their "i" index number from the word
list below. NEVER copy/reproduce the word's actual text into your
output — this applies even in English, and is mandatory for non-Latin
scripts. Only "label_text" fields (which you write yourself, in
whatever language fits) contain actual text; all word REFERENCES are
index numbers only.

TYPE "label": a single callout box for ONE concrete concept.
  Fields: "type": "label", "anchor_index" (int, from the word list),
  "label_text" (2-4 words, your own text), "icon" (see icon list below).

TYPE "comparison": TWO related concepts the script explicitly
  contrasts or equates (e.g. "AI Agent is like an employee, Agentic AI
  is like the whole team"). Only use this when the script actually
  draws a direct comparison between two named things — don't invent one.
  Fields: "type": "comparison", "anchor_index" (int, where the
  comparison begins),
  "left": {{"label_text": "...", "icon": "..."}},
  "right": {{"label_text": "...", "icon": "..."}}.

TYPE "flow": 2-4 concepts the script describes as a SEQUENCE or
  process (e.g. "it plans, then coordinates, then decides"). Only use
  this when the script actually describes an ordered sequence — don't
  invent steps that aren't there.
  Fields: "type": "flow", "anchor_index" (int, where the sequence
  begins), "num_nodes" (int, 2-4),
  "nodes": [{{"label_text": "...", "icon": "..."}}, ...] in order.

ICON LIST (use for any "icon" field): gear, arrow, box, person, brain,
chart, check, decision

STRICT SELECTION CRITERIA (applies to all types):
  - Every label_text must be a CONCRETE, SPECIFIC, NAMEABLE concept
    actually in the script — not generic verbs/connectors/filler
  - Do NOT force a comparison or flow that isn't genuinely in the
    script's content — a plain "label" is correct far more often than
    the other two types.
  - Return FEWER than {max_events} events (including zero) if the
    script doesn't have that many genuinely qualifying moments. A
    short, accurate list beats a padded, weak one.

CRITICAL — label_text MUST ADD EXPLANATORY VALUE, NOT REPEAT THE SCRIPT:
  label_text must be a SHORT, SYNTHESIZED explanation or definition of
  the concept in your own words — something a viewer learns by reading
  it, not an echo of the spoken word at that index.
    BAD (verbatim echo): word at index is "coordination" -> label_text "Coordination"
    GOOD (adds meaning):  word at index is "coordination" -> label_text "Agents share info in real time"
  If you cannot come up with a label_text that adds real explanatory
  value, DO NOT include that event at all.

CRITICAL — LANGUAGE: write every label_text (and "left"/"right" text,
and every flow node's text) in the SAME LANGUAGE the script itself is
written in. If the script is in Hindi, write label_text in Hindi. Do
NOT default to English translation — viewers reading the overlay
should see the same language they're hearing spoken.

Script (full context, for understanding meaning only — do not copy from this): {script_text}

Indexed word list (JSON) — reference these by "i" ONLY, never copy "word" text: {json.dumps(indexed_words)}

Return ONLY a JSON object: {{"events": [...]}}, where each item has a
"type" field, an "anchor_index" (or per-node indices aren't needed —
one anchor_index per event is enough, timing is computed from it in
code), plus that type's other fields. If nothing qualifies, return
{{"events": []}}.
"""

    events = []
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",  # VERIFY against console.groq.com/docs/models
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            raw = response.choices[0].message.content
            parsed = json.loads(raw)
            raw_events = parsed.get("events", []) if isinstance(parsed, dict) else parsed
            events = _resolve_events_from_indices(raw_events, word_timings)
            break  # success
        except Exception as e:
            last_error = e
            print(f"WARNING: Groq call failed on attempt {attempt + 1}/{max_retries + 1} ({e}). "
                  f"{'Retrying...' if attempt < max_retries else 'Giving up, returning zero events for this run.'}")

    if not events and last_error is not None:
        print(f"NOTE: Stage 7 overlays produced NO events for this video after "
              f"{max_retries + 1} attempts, due to: {last_error}. The base video/audio/lipsync "
              f"pipeline is unaffected — this only means no overlay graphics were added.")

    print(f"Groq selected {len(events)} event(s):")
    for e in events:
        etype = e.get("type", "label")
        if etype == "label":
            print(f"  [label] '{e.get('label_text')}' @ {e.get('start')}s")
        elif etype == "comparison":
            print(f"  [comparison] '{e.get('left', {}).get('label_text')}' vs "
                  f"'{e.get('right', {}).get('label_text')}' @ {e.get('start')}s")
        elif etype == "flow":
            nodes = " -> ".join(n.get("label_text", "?") for n in e.get("nodes", []))
            print(f"  [flow] {nodes} @ {e.get('start')}s")

    return events


_font_cache = {}


def _measure_text_width(text: str, font: ImageFont.FreeTypeFont) -> int:
    """
    Measures the actual rendered pixel width of text in the given font.
    Uses a throwaway 1x1 ImageDraw purely for measurement — no visible
    image involved. This is what render_label_frame()/render_comparison_frame()
    use to size their canvas to fit the real text instead of guessing a
    fixed width and clipping longer strings.
    """
    dummy_img = Image.new("RGBA", (1, 1))
    dummy_draw = ImageDraw.Draw(dummy_img)
    return int(dummy_draw.textlength(text, font=font))


def _contains_devanagari(text: str) -> bool:
    """Checks for any character in the Devanagari Unicode block (U+0900-U+097F),
    used for Hindi and several other Indic languages."""
    return any(0x0900 <= ord(ch) <= 0x097F for ch in text)


def _get_font(size: int, text: str = "") -> ImageFont.FreeTypeFont:
    """
    Loads a bold TrueType font at the given size. text is used to
    detect script and pick an appropriate font — DejaVu Sans (this
    file's default, bundled with matplotlib) does NOT include
    Devanagari glyphs, which would make Hindi label_text render as
    empty/invisible boxes even though the underlying string is
    correct — the same visible symptom as the original missing-font
    bug, but caused by missing script coverage instead of a missing
    font file entirely.
    """
    cache_key = (size, _contains_devanagari(text))
    if cache_key in _font_cache:
        return _font_cache[cache_key]

    font = None

    if _contains_devanagari(text):
        # VERIFY these paths against your actual pod image — font
        # package names/locations vary by distro. setup_runpod.sh
        # installs fonts-noto-core, which should provide the first path.
        devanagari_candidates = [
            "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Bold.ttf",
            "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
            "/usr/share/fonts/truetype/lohit-devanagari/Lohit-Devanagari.ttf",
        ]
        for candidate in devanagari_candidates:
            if Path(candidate).exists():
                font = ImageFont.truetype(candidate, size)
                print(f"Devanagari font loaded from: {candidate}")
                break
        if font is None:
            print("WARNING: Hindi/Devanagari text detected but NO Devanagari-capable "
                  "font was found on this system. Text will render as empty boxes. "
                  "Run: apt-get install -y fonts-noto-core  (see setup_runpod.sh)")
            # Fall through to the Latin font search below as a last resort —
            # it will NOT render Devanagari correctly, but at least won't crash.

    if font is None:
        try:
            import matplotlib
            bundled_font_path = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans-Bold.ttf"
            if bundled_font_path.exists():
                font = ImageFont.truetype(str(bundled_font_path), size)
        except Exception as e:
            print(f"WARNING: matplotlib font lookup failed ({e})")

    if font is None:
        for candidate in [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        ]:
            if Path(candidate).exists():
                font = ImageFont.truetype(candidate, size)
                break

    if font is None:
        print("WARNING: no TrueType font found anywhere — falling back to "
              "Pillow's tiny default bitmap font. Text WILL be illegible/"
              "invisible after animation scaling.")
        font = ImageFont.load_default()

    _font_cache[cache_key] = font
    return font


def _draw_icon(draw: ImageDraw.Draw, cx: int, cy: int, size: int, icon: str, color=(255, 255, 255)):
    """
    Draws a simple geometric icon centered at (cx, cy). These are
    intentionally basic shapes, not a real icon library — swap this
    function out for actual PNG icon assets once you have a design
    pass done; this exists so Stage 7 doesn't depend on any external
    icon files to run at all.
    """
    r = size // 2
    if icon == "gear":
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, width=3)
        draw.ellipse([cx - r // 2, cy - r // 2, cx + r // 2, cy + r // 2], outline=color, width=2)
    elif icon == "arrow":
        draw.line([cx - r, cy, cx + r, cy], fill=color, width=3)
        draw.line([cx + r // 2, cy - r // 2, cx + r, cy], fill=color, width=3)
        draw.line([cx + r // 2, cy + r // 2, cx + r, cy], fill=color, width=3)
    elif icon == "box":
        draw.rectangle([cx - r, cy - r, cx + r, cy + r], outline=color, width=3)
    elif icon == "person":
        draw.ellipse([cx - r // 2, cy - r, cx + r // 2, cy], outline=color, width=3)
        draw.arc([cx - r, cy - r // 2, cx + r, cy + r * 1.5], start=200, end=340, fill=color, width=3)
    elif icon == "brain":
        draw.ellipse([cx - r, cy - r // 2, cx + r, cy + r // 2], outline=color, width=3)
        draw.line([cx, cy - r // 2, cx, cy + r // 2], fill=color, width=2)
    elif icon == "chart":
        draw.line([cx - r, cy + r, cx - r, cy - r // 2], fill=color, width=3)
        draw.line([cx - r, cy + r, cx + r, cy + r], fill=color, width=3)
        draw.line([cx - r // 2, cy + r, cx - r // 2, cy], fill=color, width=3)
        draw.line([cx, cy + r, cx, cy - r // 3], fill=color, width=3)
        draw.line([cx + r // 2, cy + r, cx + r // 2, cy - r], fill=color, width=3)
    elif icon == "check":
        draw.line([cx - r, cy, cx - r // 4, cy + r // 2], fill=color, width=4)
        draw.line([cx - r // 4, cy + r // 2, cx + r, cy - r // 2], fill=color, width=4)
    elif icon == "decision":
        draw.polygon([(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)], outline=color, width=3)
    else:
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, width=3)  # fallback dot/circle


_face_cascade = None


def _get_face_cascade():
    global _face_cascade
    if _face_cascade is None:
        # Bundled with opencv-python itself — same "use what's guaranteed
        # to already be installed" reasoning as the matplotlib font fix,
        # no separate model download needed.
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _face_cascade = cv2.CascadeClassifier(cascade_path)
    return _face_cascade


def analyze_frame_for_overlay(frame: np.ndarray, box_height_px: int = 110,
                               box_width_px: int = 420) -> dict:
    """
    Looks at an ACTUAL video frame (not a fixed assumption) and decides:
      1. WHERE there's empty space to place an overlay, avoiding the
         detected face — checked per-event, so placement can move
         around the frame across the video rather than staying pinned
         to one spot the whole time.
      2. Whether that region's background is light or dark, so the
         overlay's colors can be chosen to stay readable against it.

    Returns {"y_frac": float, "x_frac": float, "placement": str,
             "is_light_bg": bool, "face_found": bool, "safe_to_place": bool}
    placement is one of "above", "side_left", "side_right", "default" (no face found).

    REVISION HISTORY:
    - Landing on the mouth (close-up shot, "below face" fallback) ->
      removed the below-face fallback, ABOVE-only.
    - That made EVERY event get skipped on a genuinely tight close-up
      video (confirmed from your own log: 4/4 events skipped, 0
      overlays placed) — above-only was too conservative for footage
      where the head fills most of the frame vertically but there's
      still visible background to the LEFT and RIGHT of the face (your
      screenshots show a blurred cafe scene on both sides). THIS
      REVISION adds side placement as a second option: if there isn't
      room above the head, check for room to the left or right of the
      face and use whichever side is wider. Only returns
      safe_to_place=False if NONE of above/left/right have room — a
      stricter bar than before, but not impossibly strict.

    box_height_px / box_width_px: the actual pixel size of the box
    you're about to place — needed for BOTH the vertical (above) and
    horizontal (side) clearance checks now.

    HEAD_CLEARANCE NOTE: Haar cascade face detection bounds roughly
    eyebrows-to-chin, NOT hair or forehead. HEAD_CLEARANCE_FRAC adds an
    empirical buffer for hair/forehead space Haar doesn't know about.
    This is an estimate, not a measurement — VERIFY against your own
    footage.

    HONESTY NOTE: Haar cascade face detection is fast and dependency-
    free, but older/less accurate than modern DNN-based detectors —
    expect occasional missed detections, especially close-up or at an
    angle.
    """
    h, w = frame.shape[0], frame.shape[1]
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = _get_face_cascade().detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    HEAD_CLEARANCE_FRAC = 0.14  # estimated hair/forehead space Haar's face bbox excludes — VERIFY
    box_margin_frac_v = (box_height_px + 30) / h
    total_clearance_v = HEAD_CLEARANCE_FRAC + box_margin_frac_v
    SIDE_MARGIN_PX = 20  # breathing room between the box and the face's edge, horizontally

    if len(faces) == 0:
        # No face detected — default to a safe upper placement, but
        # flag face_found=False so callers know this is an assumption,
        # not a measurement (could be a genuinely wide shot, or a
        # close-up so tight even Haar can't find a face).
        candidate_y_frac, candidate_x_frac = 0.06, 0.5
        placement = "default"
        face_found = False
        safe_to_place = True
    else:
        fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        face_top_frac = fy / h
        space_above = face_top_frac
        space_left_px = fx
        space_right_px = w - (fx + fw)

        if space_above >= total_clearance_v:
            candidate_y_frac = max(0.03, face_top_frac - total_clearance_v)
            candidate_x_frac = 0.5
            placement = "above"
            safe_to_place = True
        elif max(space_left_px, space_right_px) >= box_width_px + SIDE_MARGIN_PX:
            # Side placement: vertically centered on the face, on
            # whichever side has more room.
            candidate_y_frac = max(0.05, min(0.85, (fy + fh / 2) / h - (box_height_px / 2) / h))
            if space_left_px >= space_right_px:
                candidate_x_frac = (space_left_px / 2) / w  # centered within the left gap
                placement = "side_left"
            else:
                candidate_x_frac = 1.0 - (space_right_px / 2) / w  # centered within the right gap
                placement = "side_right"
            safe_to_place = True
        else:
            # No room above AND no room on either side — genuinely no
            # safe space. Skip this event rather than force a bad
            # placement.
            candidate_y_frac, candidate_x_frac = 0.03, 0.5
            placement = "none"
            safe_to_place = False
        face_found = True

    # Sample actual pixel brightness in the chosen region to decide
    # light-theme vs dark-theme colors.
    sample_y0 = int(max(0, candidate_y_frac * h))
    sample_y1 = int(min(h, sample_y0 + 100))
    sample_x0 = int(max(0, candidate_x_frac * w - 50))
    sample_x1 = int(min(w, candidate_x_frac * w + 50))
    sample_region = frame[sample_y0:sample_y1, sample_x0:sample_x1, :]
    if sample_region.size == 0:
        is_light_bg = False  # default to dark-theme styling if sampling failed
    else:
        # Standard luminance formula, averaged over the sampled region.
        luminance = (0.299 * sample_region[..., 0] + 0.587 * sample_region[..., 1]
                     + 0.114 * sample_region[..., 2]).mean()
        is_light_bg = luminance > 150  # VERIFY this threshold against your actual footage —
                                        # 150/255 is a reasonable starting guess, not a tuned value

    return {"y_frac": candidate_y_frac, "x_frac": candidate_x_frac, "placement": placement,
            "is_light_bg": is_light_bg, "face_found": face_found, "safe_to_place": safe_to_place}


def render_label_frame(label_text: str, icon: str, width: int = None, height: int = 110,
                        light_theme: bool = False, max_width: int = 900) -> np.ndarray:
    """
    Renders one overlay (rounded box + icon + text) as an RGBA numpy
    array via Pillow — no ImageMagick, no MoviePy TextClip. This static
    frame gets animated (slid/scaled in) by MoviePy afterward in
    composite_overlays(), so the MOTION comes from position/size
    functions, not from anything baked into this image.

    FIXED BUG: width used to be a fixed 420px regardless of how long
    label_text actually was, so longer text simply ran past the canvas
    edge and got clipped (visible in your screenshots — "Self-running
    team", "Single task perform...", "Automated email re..." all cut
    off). width is now None by default, meaning: measure the ACTUAL
    rendered text width first, then size the canvas to fit it (icon +
    text + padding), so the box is only as wide as it needs to be.
    Pass an explicit width to override this and force a fixed size
    instead (e.g. if you want uniform box widths across all events).

    max_width: hard cap in pixels. If measured text would exceed this,
    the font size is reduced (down to a floor of 16px) to fit rather
    than clipping — VERIFY this reads OK at the smaller size for very
    long label_text; there's no wrapping-to-multiple-lines fallback
    here, so an extremely long label at the size floor could still
    clip. Keep label_text genuinely short (2-5 words) at the prompt
    level as the real fix — this is a safety net, not a substitute for
    that.

    light_theme: controls the BOX's own color scheme, for contrast
    against the underlying video. True = light-colored box (use this
    when the video background behind the overlay is DARK, so the box
    stands out). False = dark box (use when the video background is
    LIGHT). composite_overlays() below computes this correctly as
    light_theme = NOT is_light_bg — i.e. opposite of the background's
    own brightness, since the goal is contrast, not matching.
    """
    text_x = 100  # left edge where text starts (after icon + padding)
    right_padding = 40

    font_size = 28
    font = _get_font(font_size, text=label_text)
    text_w = _measure_text_width(label_text, font)

    # Shrink font if even the max_width can't fit it, down to a floor —
    # avoids clipping for unexpectedly long label_text at the cost of
    # readability, rather than silently cutting text off.
    while (text_x + text_w + right_padding) > max_width and font_size > 16:
        font_size -= 2
        font = _get_font(font_size, text=label_text)
        text_w = _measure_text_width(label_text, font)

    if width is None:
        width = min(max(text_x + text_w + right_padding, 260), max_width)  # 260 = sane minimum box width

    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    if light_theme:
        box_fill = (245, 245, 250, 225)
        box_outline = (40, 120, 200, 255)
        text_color = (20, 20, 30, 255)
        icon_color = (40, 120, 200, 255)
    else:
        box_fill = (15, 20, 30, 220)
        box_outline = (90, 200, 255, 255)
        text_color = (255, 255, 255, 255)
        icon_color = (90, 200, 255, 255)

    draw.rounded_rectangle([0, 0, width - 1, height - 1], radius=18,
                            fill=box_fill, outline=box_outline, width=3)

    _draw_icon(draw, cx=55, cy=height // 2, size=44, icon=icon, color=icon_color)

    draw.text((text_x, height // 2), label_text, font=font, fill=text_color, anchor="lm")

    return np.array(img)


def preview_label_frame(label_text: str, icon: str = "box", save_path: str = "test_output/label_preview.png"):
    """
    Debugging helper — renders ONE label as a standalone PNG so you can
    check legibility/styling without running the full pipeline. Use
    this to iterate on font/size/color before spending time on a full
    video run. Example:

        from path_a_pipeline.pipeline.stages import stage7_overlays
        stage7_overlays.preview_label_frame("Agentic AI", icon="brain")
    """
    frame = render_label_frame(label_text, icon)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).save(save_path)
    print(f"Saved preview to {save_path} — open it directly to check before running a full video.")
    return save_path


def render_comparison_frame(left_text: str, right_text: str, left_icon: str, right_icon: str,
                             height: int = 140, light_theme: bool = False,
                             max_box_width: int = 480) -> np.ndarray:
    """
    Two boxes side by side with a small divider — matches your original
    "AI Agent = Employee" vs "Agentic AI = Team" screenshot. Rendered as
    ONE image (both boxes appear together as a unit); composite_overlays()
    animates the whole thing in as one slide/scale-in, same mechanism
    as a single label.

    FIXED BUG: this used to split a fixed 940px canvas into two equal
    halves regardless of each side's actual text length, so longer text
    (e.g. "Self-running team") got clipped by its half's boundary. Each
    box is now sized independently to its own measured text, so the two
    sides can be different widths if their content needs it — no more
    shared fixed half-width to overflow.

    light_theme: same contrast rule as render_label_frame() — True for
    a light box when placed against a dark video background, False for
    a dark box against a light background.
    """
    text_x_offset = 85  # left edge where text starts within each box, after its icon
    right_padding = 30
    gap = 60  # horizontal gap between the two boxes, holds the "VS" divider

    font_size = 24
    font = _get_font(font_size, text=left_text + right_text)
    left_text_w = _measure_text_width(left_text, font)
    right_text_w = _measure_text_width(right_text, font)

    # Shrink font (both sides together, so they stay visually consistent)
    # if either side would exceed max_box_width, down to a floor.
    while (max(left_text_w, right_text_w) + text_x_offset + right_padding) > max_box_width and font_size > 14:
        font_size -= 2
        font = _get_font(font_size, text=left_text + right_text)
        left_text_w = _measure_text_width(left_text, font)
        right_text_w = _measure_text_width(right_text, font)

    left_box_w = min(max(text_x_offset + left_text_w + right_padding, 200), max_box_width)
    right_box_w = min(max(text_x_offset + right_text_w + right_padding, 200), max_box_width)
    width = left_box_w + gap + right_box_w

    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    if light_theme:
        box_fill = (245, 245, 250, 225)
        text_color = (20, 20, 30, 255)
        left_accent = (40, 120, 200, 255)
        right_accent = (200, 100, 40, 255)
        divider_color = (80, 80, 80, 255)
    else:
        box_fill = (15, 20, 30, 220)
        text_color = (255, 255, 255, 255)
        left_accent = (90, 200, 255, 255)
        right_accent = (255, 150, 90, 255)
        divider_color = (200, 200, 200, 255)

    # Left box
    draw.rounded_rectangle([0, 0, left_box_w, height - 1], radius=18,
                            fill=box_fill, outline=left_accent, width=3)
    _draw_icon(draw, cx=45, cy=height // 2, size=40, icon=left_icon, color=left_accent)
    draw.text((text_x_offset, height // 2), left_text, font=font, fill=text_color, anchor="lm")

    # Right box
    right_x0 = left_box_w + gap
    draw.rounded_rectangle([right_x0, 0, width - 1, height - 1], radius=18,
                            fill=box_fill, outline=right_accent, width=3)
    _draw_icon(draw, cx=right_x0 + 45, cy=height // 2, size=40, icon=right_icon, color=right_accent)
    draw.text((right_x0 + text_x_offset, height // 2), right_text, font=font, fill=text_color, anchor="lm")

    # "VS" divider text, centered in the gap
    divider_font = _get_font(20)
    draw.text((left_box_w + gap // 2, height // 2), "VS", font=divider_font, fill=divider_color, anchor="mm")

    return np.array(img)


def preview_comparison_frame(left_text: str = "AI Agent", right_text: str = "Agentic AI",
                              left_icon: str = "gear", right_icon: str = "brain",
                              save_path: str = "test_output/comparison_preview.png"):
    """Debugging helper for the comparison layout — same idea as preview_label_frame()."""
    frame = render_comparison_frame(left_text, right_text, left_icon, right_icon)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).save(save_path)
    print(f"Saved preview to {save_path}")
    return save_path


def render_connector_down(width: int = 60, height: int = 50,
                           color=(90, 200, 255, 255)) -> np.ndarray:
    """Small downward arrow image, placed between two stacked flow nodes."""
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    cx = width // 2
    draw.line([cx, 5, cx, height - 15], fill=color, width=3)
    draw.polygon([(cx - 10, height - 20), (cx + 10, height - 20), (cx, height - 4)], fill=color)
    return np.array(img)


def preview_flow_frames(labels_and_icons: list = None,
                         save_dir: str = "test_output/flow_preview"):
    """
    Debugging helper for the flow layout — saves each node + connector
    as separate PNGs so you can check them individually. Default
    example matches your original "Planning -> Coordination -> Decision"
    screenshot.

        stage7_overlays.preview_flow_frames()
    """
    if labels_and_icons is None:
        labels_and_icons = [("Planning", "brain"), ("Coordination", "person"), ("Decision", "decision")]

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    for i, (label, icon) in enumerate(labels_and_icons):
        frame = render_label_frame(label, icon, width=380, height=90)
        Image.fromarray(frame).save(f"{save_dir}/node_{i}.png")
    connector = render_connector_down()
    Image.fromarray(connector).save(f"{save_dir}/connector.png")
    print(f"Saved {len(labels_and_icons)} node(s) + 1 connector to {save_dir}")
    return save_dir


def _animated_layer(frame: np.ndarray, start: float, duration: float, vid_h: int, vid_w: int,
                     target_y_frac: float = 0.15, target_x_frac: float = 0.5,
                     anim_in: float = 0.35, anim_out: float = 0.25):
    """
    Shared animation wrapper — slide-up + scale-in entrance, fade-out
    exit. Used by all three event types so label/comparison/flow all
    move consistently. Factored out of REV 2's composite_overlays()
    inline logic so it's reusable per-node for flow diagrams too.

    target_x_frac added this revision: horizontal position was
    hardcoded to "center" before, which only worked for above-head
    placement. Side placement (analyze_frame_for_overlay's new
    "side_left"/"side_right" option) needs to position off-center —
    target_x_frac=0.5 keeps the old centered behavior as the default,
    so existing "above" placement calls are unaffected.
    """
    h, w = frame.shape[0], frame.shape[1]

    def pos(t):
        if t < anim_in:
            progress = t / anim_in
            y_offset = (1 - progress) * 40
        else:
            y_offset = 0
        x_pos = vid_w * target_x_frac - w / 2  # convert center-fraction to a left-edge pixel position
        return (x_pos, vid_h * target_y_frac + y_offset)

    def size_at(t):
        if t < anim_in:
            progress = t / anim_in
            scale = 0.7 + 0.3 * progress
        else:
            scale = 1.0
        return (int(w * scale), int(h * scale))

    return (ImageClip(frame)
            .set_start(start)
            .set_duration(duration)
            .set_position(pos)
            .resize(size_at)
            .crossfadeout(anim_out))


def composite_overlays(video_path: Path, callout_events: list,
                        output_path: Path) -> dict:
    """
    Layers animated graphics onto video_path. Handles all three event
    types:
      "label"      — one animated box
      "comparison" — one animated two-box image (whole thing enters as a unit)
      "flow"       — nodes stacked vertically, each entering in sequence
                     with a connector arrow between consecutive nodes

    PLACEMENT + COLOR ARE PER-EVENT AND VIDEO-AWARE ACROSS THE WHOLE
    EVENT WINDOW, NOT JUST ITS START FRAME:
    For each event, analyze_region_for_overlay() samples several frames
    across [event["start"], event["end"]] (not only the start moment),
    tracks the face's UNION bounding box across that whole window, and
    picks whichever safe candidate region (above/left/right) is
    visually quietest by edge-density — so placement stays clear of the
    subject for the entire time the overlay is on screen, even if they
    move during it, and tends to land somewhere that won't visually
    compete with a busy background.

    RENDERING IS NOW A REAL PNG FRAME SEQUENCE PER EVENT, NOT ONE
    STATIC IMAGE PUSHED THROUGH AN AFFINE TRANSFORM:
    render_animated_label_frames()/render_animated_comparison_frames()
    render each card as HTML/CSS (glassmorphism, layered shadow,
    gradient icon tile, cubic-bezier entrance easing, per-word stagger)
    via headless Chromium, with the exit fade baked directly into the
    frames' alpha channel. build_moviepy_layer() turns that sequence
    into a positioned MoviePy clip with a real per-frame alpha mask —
    this replaces the old _animated_layer() slide/scale wrapper
    entirely for label/comparison/flow-node events.
    """
    base = VideoFileClip(str(video_path))
    layers = [base]
    placed_events = []  # tracks what was ACTUALLY placed, separate from what was attempted —
                         # safe_to_place=False events get skipped and shouldn't be counted as placed

    NODE_STAGGER = 0.7  # seconds between each flow node's entrance
    FLOW_NODE_HEIGHT = 90
    FLOW_NODE_GAP = 20  # vertical gap between stacked nodes, holds the connector arrow
    FADE_OUT_S = 0.25   # baked into each rendered frame sequence's alpha, not a MoviePy transform anymore

    for event_idx, event in enumerate(callout_events):
        etype = event.get("type", "label")
        # Per-event output dir for rendered frames — avoids overwriting
        # the previous event's frames with a shared default path, and
        # means you can inspect any event's intermediate PNGs after a
        # run (e.g. test_output/_overlay_frames/event_2_label/f0012.png).
        event_out_dir = f"test_output/_overlay_frames/event_{event_idx}_{etype}"

        # Box height/width vary by type — passed in so BOTH the
        # vertical (above-head) and horizontal (side) clearance checks
        # scale correctly. For "flow", use the FULL STACK height (all
        # nodes + gaps), not just one node.
        if etype == "flow":
            num_nodes = len(event.get("nodes", []))
            expected_box_height = max(num_nodes * (FLOW_NODE_HEIGHT + FLOW_NODE_GAP) - FLOW_NODE_GAP, FLOW_NODE_HEIGHT)
            expected_box_width = 380
        elif etype == "comparison":
            expected_box_height, expected_box_width = 140, 480  # approx — comparison width is dynamic per-text,
                                                                   # this is a reasonable upper-bound estimate for
                                                                   # the clearance check, not the final render size
        else:
            expected_box_height, expected_box_width = 110, 420  # approx — label width is also dynamic per-text

        # Tracks the face across the event's full [start, end] window
        # (not just its start frame) and scores candidate regions by
        # visual busyness — this is what makes placement stay safe for
        # the whole time the overlay is visible, and tends to avoid
        # cluttered backgrounds rather than always trying "above" first.
        analysis = analyze_region_for_overlay(
            base.get_frame, event["start"], event["end"],
            box_width_px=expected_box_width, box_height_px=expected_box_height,
        )

        if not analysis["safe_to_place"]:
            print(f"SKIPPED event at {event['start']}s "
                  f"('{event.get('label_text', event.get('nodes', '?'))}') "
                  f"— no safe space above the head OR to either side across this event's duration. "
                  f"Better to omit than risk covering the mouth.")
            continue

        y_frac = analysis["y_frac"]
        x_frac = analysis["x_frac"]
        light_theme = not analysis["is_light_bg"]  # contrast: light box on dark bg, dark box on light bg
        if not analysis["face_found"]:
            print(f"NOTE: no face detected across {event['start']}-{event['end']}s — using default "
                  f"placement/theme, verify this visually since the fallback is a guess, not a measurement.")
        else:
            print(f"  placement for event @ {event['start']}s: {analysis['placement']} "
                  f"(busyness={analysis.get('busyness_score')}, "
                  f"{analysis.get('candidates_considered')} safe candidate(s) considered)")
        placed_events.append(event)

        if etype == "label":
            duration = event["end"] - event["start"]
            if duration <= 0:
                continue
            frame_paths, fw, fh = render_animated_label_frames(
                event["label_text"], event.get("icon", "box"), fps=base.fps, duration_s=duration,
                light_theme=light_theme, fade_out_s=min(FADE_OUT_S, duration / 4), out_dir=event_out_dir,
            )
            layers.append(build_moviepy_layer(frame_paths, base.fps, event["start"], x_frac, y_frac, base.w, base.h))

        elif etype == "comparison":
            duration = event["end"] - event["start"]
            if duration <= 0:
                continue
            left = event.get("left", {})
            right = event.get("right", {})
            frame_paths, fw, fh = render_animated_comparison_frames(
                left.get("label_text", "?"), right.get("label_text", "?"),
                left.get("icon", "box"), right.get("icon", "box"),
                fps=base.fps, duration_s=duration, light_theme=light_theme,
                fade_out_s=min(FADE_OUT_S, duration / 4), out_dir=event_out_dir,
            )
            layers.append(build_moviepy_layer(frame_paths, base.fps, event["start"], x_frac, y_frac, base.w, base.h))

        elif etype == "flow":
            nodes = event.get("nodes", [])
            group_start = event["start"]
            group_end = event["end"]
            if not nodes or group_end <= group_start:
                continue

            # Vertical stack starting at the per-event detected y_frac/
            # x_frac (not hardcoded constants), each node entering
            # NODE_STAGGER seconds after the previous one, all holding
            # until the group's overall end time. Uses ONE placement
            # analysis for the whole group (across the group's full
            # [start, end] window) rather than re-analyzing per node —
            # a deliberate simplification.
            for i, node in enumerate(nodes):
                node_start = group_start + i * NODE_STAGGER
                node_duration = max(group_end - node_start, 0.3)
                frame_paths, fw, fh = render_animated_label_frames(
                    node.get("label_text", "?"), node.get("icon", "box"), fps=base.fps,
                    duration_s=node_duration, light_theme=light_theme,
                    fade_out_s=min(FADE_OUT_S, node_duration / 4), out_dir=f"{event_out_dir}/node_{i}",
                )
                node_y_frac = y_frac + i * (FLOW_NODE_HEIGHT + FLOW_NODE_GAP) / base.h
                layers.append(build_moviepy_layer(frame_paths, base.fps, node_start, x_frac, node_y_frac,
                                                    base.w, base.h))

                # Connector arrow appears right as the NEXT node starts,
                # sitting in the gap between this node and the next one.
                # Kept on the original Pillow/_animated_layer path — a
                # small static arrow doesn't benefit from the HTML/CSS
                # upgrade the way text cards do, and reusing the tested
                # legacy renderer here avoids adding render surface
                # area for no visible gain.
                if i < len(nodes) - 1:
                    connector_frame = render_connector_down(
                        color=(40, 120, 200, 255) if light_theme else (90, 200, 255, 255))
                    connector_start = group_start + (i + 1) * NODE_STAGGER
                    connector_duration = max(group_end - connector_start, 0.3)
                    connector_y_frac = y_frac + i * (FLOW_NODE_HEIGHT + FLOW_NODE_GAP) / base.h \
                        + FLOW_NODE_HEIGHT / base.h
                    layers.append(_animated_layer(connector_frame, connector_start, connector_duration,
                                                   base.h, base.w, target_y_frac=connector_y_frac,
                                                   target_x_frac=x_frac,
                                                   anim_in=0.15))  # connectors pop in faster than nodes

        else:
            print(f"WARNING: unknown event type '{etype}', skipping this event: {event}")

    final = CompositeVideoClip(layers)
    final = final.set_duration(base.duration)

    # STRATEGY CHANGE: two targeted MoviePy audio/fps fixes didn't move
    # the glitch at all — same symptom, same exact spot, both times.
    # That's a strong signal this is a MoviePy CompositeVideoClip audio
    # internals bug, not something fixable by tuning more of its
    # parameters. New approach: render the composited video SILENTLY
    # via MoviePy (video/overlay compositing is what MoviePy is doing
    # correctly here — it's specifically the audio path that's
    # suspect), then mux the ORIGINAL audio back on with a direct
    # FFmpeg call — the same proven pattern stage6_export.py already
    # uses successfully elsewhere in this pipeline, bypassing MoviePy's
    # audio handling entirely.
    silent_video_path = output_path.parent / f"_silent_{output_path.name}"
    final.write_videofile(str(silent_video_path), codec="libx264", audio=False, fps=base.fps, logger=None)
    base.close()
    final.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mux_cmd = [
        "ffmpeg", "-y",
        "-i", str(silent_video_path),
        "-i", str(video_path),  # original video_path still has its original audio track
        "-map", "0:v",          # video from the silent composited render
        "-map", "1:a",          # audio from the ORIGINAL source, untouched by MoviePy
        "-c:v", "copy",         # no re-encoding of the video we just rendered
        "-c:a", "aac",
        "-shortest",
        str(output_path),
    ]
    result = subprocess.run(mux_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg audio mux failed:\n{result.stderr}")

    silent_video_path.unlink(missing_ok=True)  # clean up the intermediate silent file

    # VERIFY: confirm final duration actually matches the original —
    # this now depends on FFmpeg's mux, not MoviePy's audio inference.
    verify = VideoFileClip(str(output_path))
    print(f"VERIFY: original video duration vs final output duration={verify.duration}s "
          f"— compare this against your source video's real duration.")
    verify.close()

    return {"num_events": len(placed_events), "num_attempted": len(callout_events), "events": placed_events}


def preview_placement_on_video(video_path: str, timestamps: list = None,
                                save_dir: str = "test_output/placement_preview"):
    """
    Debugging helper — checks analyze_frame_for_overlay() against your
    ACTUAL video at a few real timestamps, without running the LLM or
    the full compositing pipeline. Saves each sampled frame with a
    rectangle drawn where the overlay WOULD be placed, so you can
    visually confirm face-avoidance and light/dark detection are
    working before trusting them in a real run.

    NOTE: this checks single instants, using the older single-frame
    analyze_frame_for_overlay() — it's for a quick eyeball check of a
    few specific moments. The real pipeline (composite_overlays())
    uses analyze_region_for_overlay() instead, which tracks the face
    across each event's full [start, end] window rather than one
    instant. For a closer approximation of real placement behavior at
    a given moment, pass a timestamp near the middle of where you'd
    expect an event to land, or just run a full preview via
    composite_overlays() on a short test clip.

        stage7_overlays.preview_placement_on_video(
            "/content/my_video.mp4", timestamps=[1.0, 5.0, 10.0]
        )
    """
    if timestamps is None:
        timestamps = [1.0, 5.0, 10.0]

    clip = VideoFileClip(str(video_path))
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    for t in timestamps:
        if t > clip.duration:
            print(f"Skipping t={t}s — video is only {clip.duration:.1f}s long")
            continue
        frame = clip.get_frame(t)
        analysis = analyze_frame_for_overlay(frame)

        annotated = frame.copy()
        h, w = annotated.shape[0], annotated.shape[1]

        if not analysis["safe_to_place"]:
            # Yellow full-frame border = this event would be SKIPPED
            # entirely in a real run, not placed anywhere.
            cv2.rectangle(annotated, (0, 0), (w - 1, h - 1), (255, 255, 0), 8)
            cv2.putText(annotated, "UNSAFE - would be SKIPPED", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        else:
            y0 = int(analysis["y_frac"] * h)
            x_center = int(analysis["x_frac"] * w)
            box_w_preview = 300  # just for drawing the preview rectangle, not the real render width
            x0 = max(0, x_center - box_w_preview // 2)
            x1 = min(w, x_center + box_w_preview // 2)
            color = (0, 255, 0) if analysis["face_found"] else (255, 0, 0)  # green=face found, red=fallback used
            cv2.rectangle(annotated, (x0, y0), (x1, y0 + 100), color, 4)
            theme_label = "DARK-BG->light box" if not analysis["is_light_bg"] else "LIGHT-BG->dark box"
            label = f"{analysis['placement']} | {theme_label}"
            cv2.putText(annotated, label, (x0, max(30, y0 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        out_path = f"{save_dir}/t{t}.png"
        Image.fromarray(annotated).save(out_path)
        print(f"t={t}s: safe_to_place={analysis['safe_to_place']}, placement={analysis['placement']}, "
              f"face_found={analysis['face_found']}, y_frac={analysis['y_frac']:.2f}, "
              f"x_frac={analysis['x_frac']:.2f}, "
              f"is_light_bg={analysis['is_light_bg']} -> saved {out_path}")

    clip.close()
    print(f"\nCheck the saved frames in {save_dir} — the colored rectangle shows where an "
          f"overlay would be placed, and the text shows which color theme would be used.")
    return save_dir


def run_overlays(video_path: Path, audio_path: Path, script_text: str,
                  output_path: Path, max_events: int = 5) -> dict:
    """Full Stage 7: timings -> Groq event detection -> animated compositing."""
    word_timings = extract_word_timings(audio_path)
    callout_events = detect_callout_events(script_text, word_timings, max_events=max_events)
    composite_result = composite_overlays(video_path, callout_events, output_path)

    return {
        "num_words_transcribed": len(word_timings),
        **composite_result,
        "output_path": str(output_path),
    }


if __name__ == "__main__":
    import sys
    import time

    if len(sys.argv) < 4:
        print("Usage: python stage7_overlays.py <video_path> <audio_path> <script_text_file>")
        sys.exit(1)

    video_path = Path(sys.argv[1])
    audio_path = Path(sys.argv[2])
    script_text = Path(sys.argv[3]).read_text()
    out_path = Path("test_output/07_with_overlays.mp4")

    start = time.time()
    params = run_overlays(video_path, audio_path, script_text, out_path)
    elapsed = time.time() - start

    print(f"\nOverlay compositing done in {elapsed:.1f}s -> {out_path}")
    print(f"Events placed: {params['num_events']}")
    print("CHECKPOINT: watch the full video. Check for:")
    print("  - overlays landing at the RIGHT moment, not drifted off from the audio")
    print("  - real eased entrance + per-word reveal + fade-out, not a static box appearing")
    print("  - LLM-selected moments are genuinely relevant, and comparison/flow types")
    print("    are only used when the script ACTUALLY has that structure")
    print("  - overlays stay clear of the face for their WHOLE on-screen duration, not just")
    print("    the moment they appear — placement now tracks the face across each event's")
    print("    full window via analyze_region_for_overlay(), but it's still Haar cascade")
    print("    under the hood in this environment (see overlay_placement.py's top-of-file")
    print("    note on swapping in MediaPipe where you have full internet access) and can")
    print("    still miss faces at an angle; check this visually rather than assuming it's right")
    print()
    print("TIP: before running this on a full video again, use the cheap preview helpers:")
    print("  stage7_overlays.preview_label_frame('Agentic AI', icon='brain')  # legacy Pillow preview")
    print("  stage7_overlays.preview_comparison_frame()                       # legacy Pillow preview")
    print("  stage7_overlays.preview_flow_frames()                            # legacy Pillow preview")
    print("  from .overlay_render import render_animated_label_frames         # new HTML/CSS renderer —")
    print("    render_animated_label_frames('Agentic AI', 'brain', fps=24, duration_s=2.0)")
    print("    then open a frame from the returned paths directly to check styling")
