"""
Animated overlay renderer: builds each event as an HTML/CSS card (same
glassmorphism/shadow/gradient treatment from the comparison), then scrubs
a real CSS animation via headless Chromium (Playwright) and captures a
PNG per output frame. This replaces the single affine-transform frame
that _animated_layer() used to slide/scale — the entrance now has actual
cubic-bezier easing and a per-word stagger on the text.

Also handles TEXT PLACEMENT better than the original: instead of only
shrinking font size to force one line, it wraps to a second line once
the label would exceed a preferred max width, and the box always
clamps within a safe margin of the frame edges.
"""
from playwright.sync_api import sync_playwright
from PIL import Image
import io, textwrap, concurrent.futures

_TEMPLATE = """
<!DOCTYPE html><html><head><style>
  html,body {{ margin:0; padding:0; background:transparent; }}
  .card {{
    display:inline-flex; align-items:center; gap:16px;
    padding:16px 28px 16px 18px; border-radius:20px;
    background:linear-gradient(135deg, rgba(255,255,255,{bg_alpha}), rgba(255,255,255,{bg_alpha2}));
    backdrop-filter:blur(18px) saturate(160%);
    -webkit-backdrop-filter:blur(18px) saturate(160%);
    border:1px solid rgba(255,255,255,0.65);
    box-shadow:0 8px 24px rgba(0,0,0,0.28),0 2px 6px rgba(0,0,0,0.18),inset 0 1px 0 rgba(255,255,255,0.6);
    font-family:-apple-system,"Segoe UI","Helvetica Neue",Arial,sans-serif;
    opacity:0; transform:translateY(14px) scale(0.9);
    animation:enter 0.55s cubic-bezier(0.22,1,0.36,1) forwards;
  }}
  @keyframes enter {{
    0%   {{ opacity:0; transform:translateY(14px) scale(0.9); }}
    70%  {{ opacity:1; transform:translateY(-3px) scale(1.02); }}
    100% {{ opacity:1; transform:translateY(0) scale(1); }}
  }}
  .icon-wrap {{
    width:44px;height:44px;border-radius:12px;flex-shrink:0;
    background:linear-gradient(135deg,{icon_c1},{icon_c2});
    display:flex;align-items:center;justify-content:center;
    box-shadow:0 3px 8px rgba(90,80,220,0.45);
  }}
  .icon-wrap svg {{ width:24px;height:24px; }}
  .label {{ font-size:26px; font-weight:650; letter-spacing:-0.2px; color:{text_color};
            text-shadow:0 1px 0 rgba(255,255,255,0.25); line-height:1.15; max-width:{max_text_w}px; }}
  .word {{ display:inline-block; opacity:0; filter:blur(3px);
           animation:word-in 0.4s cubic-bezier(0.22,1,0.36,1) forwards; }}
  @keyframes word-in {{ to {{ opacity:1; filter:blur(0); }} }}
</style></head><body>
  <div class="card" id="card">
    <div class="icon-wrap">{icon_svg}</div>
    <div class="label">{words_html}</div>
  </div>
</body></html>
"""

_ICONS = {
    "brain": '<svg viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9.5 2a2.5 2.5 0 0 0-2.4 3.2A3 3 0 0 0 5 8v1a3 3 0 0 0-1 5.8 2.5 2.5 0 0 0 2.4 3.2 2.5 2.5 0 0 0 4.6 1H9.5A2.5 2.5 0 0 1 7 16.5v-9A2.5 2.5 0 0 1 9.5 5z"/><path d="M14.5 2a2.5 2.5 0 0 1 2.4 3.2A3 3 0 0 1 19 8v1a3 3 0 0 1 1 5.8 2.5 2.5 0 0 1-2.4 3.2 2.5 2.5 0 0 1-4.6 1h.5a2.5 2.5 0 0 0 2.5-2.5v-9A2.5 2.5 0 0 0 14.5 5z"/></svg>',
    "chart": '<svg viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><path d="M18 17V9"/><path d="M13 17V5"/><path d="M8 17v-3"/></svg>',
    "check": '<svg viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>',
}

MAX_LABEL_WIDTH_PX = 260  # wrap to 2 lines before this, rather than shrinking font indefinitely


def _wrap_label(text: str) -> str:
    """Wraps long labels onto two lines by word count rather than
    letting a single line run arbitrarily wide (the original approach
    just shrank font size, which makes long labels illegible)."""
    if len(text) <= 16:
        return text
    wrapped = textwrap.wrap(text, width=max(10, len(text) // 2 + 2))
    return "<br>".join(wrapped[:2])


def _words_html(text: str, stagger_step=0.06, base_delay=0.15) -> str:
    wrapped = _wrap_label(text)
    parts = []
    i = 0
    for line in wrapped.split("<br>"):
        for word in line.split(" "):
            delay = base_delay + i * stagger_step
            parts.append(f'<span class="word" style="animation-delay:{delay:.2f}s">{word}</span>')
            i += 1
        parts.append("<br>")
    if parts and parts[-1] == "<br>":
        parts.pop()
    return " ".join(parts)


def _apply_fade_tail(png_path: str, t: float, duration_s: float, fade_out_s: float):
    """Multiplies the frame's alpha channel down to 0 over the last
    fade_out_s seconds of duration_s. Done as a direct pixel op on the
    already-rendered PNG rather than via CSS, since the CSS keyframe
    only needs to describe the entrance — this keeps the exit fade
    exact regardless of how long the "hold" period between entrance
    and exit ends up being for a given event's duration."""
    if fade_out_s <= 0:
        return
    fade_start = duration_s - fade_out_s
    if t < fade_start:
        return
    factor = max(0.0, 1.0 - (t - fade_start) / fade_out_s)
    img = Image.open(png_path).convert("RGBA")
    r, g, b, a = img.split()
    a = a.point(lambda v: int(v * factor))
    Image.merge("RGBA", (r, g, b, a)).save(png_path)


def _run_isolated(fn, *args, **kwargs):
    """
    Runs fn in a brand-new thread with no asyncio event loop of its own.

    WHY THIS EXISTS: Playwright's sync API explicitly refuses to run
    inside a thread that already has a running asyncio event loop —
    it raises 'It looks like you are using Playwright Sync API inside
    the asyncio loop' rather than silently misbehaving. Colab (and
    Jupyter generally) runs its own event loop under the hood for
    every cell, so calling render_animated_label_frames() directly
    from a notebook cell hits this every time, even though the exact
    same code runs fine from a plain `python stage7_overlays.py ...`
    script (no event loop there). Running the Playwright call in a
    fresh thread sidesteps this everywhere, at the cost of a small
    thread-creation overhead per event — negligible next to the
    browser launch + screenshot work itself.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(fn, *args, **kwargs).result()


def render_animated_label_frames(label_text: str, icon: str, fps: int, duration_s: float,
                                  light_theme: bool = False, out_dir: str = "anim_frames",
                                  fade_out_s: float = 0.25):
    """Public entry point — see _render_animated_label_frames_sync() for
    the actual implementation. Runs it in an isolated thread so this is
    safe to call from Colab/Jupyter (running event loop) as well as
    plain scripts (no event loop)."""
    return _run_isolated(_render_animated_label_frames_sync, label_text, icon, fps, duration_s,
                          light_theme, out_dir, fade_out_s)


def _render_animated_label_frames_sync(label_text: str, icon: str, fps: int, duration_s: float,
                                        light_theme: bool = False, out_dir: str = "anim_frames",
                                        fade_out_s: float = 0.25):
    """Renders the card's full on-screen lifetime (entrance, per the CSS
    @keyframes; hold; then a python-side alpha fade-out for the last
    fade_out_s seconds) as a PNG sequence, one file per output frame.
    Returns (frame_paths, width, height)."""
    import os
    os.makedirs(out_dir, exist_ok=True)

    if light_theme:
        bg_alpha, bg_alpha2, text_color = "0.55", "0.28", "#16182a"
    else:
        bg_alpha, bg_alpha2, text_color = "0.18", "0.10", "#f2f4ff"

    html = _TEMPLATE.format(
        bg_alpha=bg_alpha, bg_alpha2=bg_alpha2, text_color=text_color,
        icon_c1="#6a8dff", icon_c2="#7b5cff",
        icon_svg=_ICONS.get(icon, _ICONS["check"]),
        words_html=_words_html(label_text), max_text_w=MAX_LABEL_WIDTH_PX,
    )
    html_path = f"{out_dir}/_card.html"
    with open(html_path, "w") as f:
        f.write(html)

    n_frames = max(1, int(fps * duration_s))
    frame_paths = []
    PAD = 24  # extra margin around the settled box so the slide-in offset
              # and scale-overshoot don't get clipped at the frame edges
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 700, "height": 220})
        page.goto(f"file://{os.path.abspath(html_path)}")
        card = page.query_selector("#card")

        # BUG FIX: element.screenshot() captures the CURRENT visual
        # bounding box, which the entrance animation's `transform:
        # scale(...)` shrinks/grows frame to frame (90% -> 102% -> 100%)
        # — each screenshot came out a slightly different pixel size,
        # which MoviePy's ImageSequenceClip then rejected outright.
        # Fix: scrub well past the end of any animation to measure the
        # SETTLED (scale=1) box once, then use that same fixed clip
        # region for every single frame regardless of t.
        page.evaluate(
            """(t) => { document.getAnimations().forEach(a => { a.pause(); a.currentTime = t; }); }""",
            (duration_s + 1) * 1000,
        )
        settled = card.bounding_box()
        clip_rect = {
            "x": max(0, settled["x"] - PAD), "y": max(0, settled["y"] - PAD),
            "width": settled["width"] + 2 * PAD, "height": settled["height"] + 2 * PAD,
        }
        box = (int(clip_rect["width"]), int(clip_rect["height"]))

        for i in range(n_frames):
            t = i / fps
            # scrub every animation on the page to time t (ms), paused —
            # entrance keyframe finishes early and holds its end state
            # (animation-fill-mode: forwards) for any t past it, which
            # is exactly the "hold" phase we want.
            page.evaluate(
                """(t) => { document.getAnimations().forEach(a => { a.pause(); a.currentTime = t; }); }""",
                t * 1000,
            )
            path = f"{out_dir}/f{i:04d}.png"
            page.screenshot(path=path, clip=clip_rect, omit_background=True)
            _apply_fade_tail(path, t, duration_s, fade_out_s)
            frame_paths.append(path)
        browser.close()

    return frame_paths, box[0], box[1]


_COMPARISON_TEMPLATE = """
<!DOCTYPE html><html><head><style>
  html,body {{ margin:0; padding:0; background:transparent; }}
  .wrap {{ display:flex; align-items:center; gap:18px; font-family:-apple-system,"Segoe UI",Arial,sans-serif; }}
  .card {{
    display:flex; flex-direction:column; align-items:flex-start; gap:8px;
    padding:16px 24px; border-radius:20px;
    background:linear-gradient(135deg, rgba(255,255,255,{bg_alpha}), rgba(255,255,255,{bg_alpha2}));
    backdrop-filter:blur(18px) saturate(160%); -webkit-backdrop-filter:blur(18px) saturate(160%);
    border:1px solid rgba(255,255,255,0.65);
    box-shadow:0 8px 24px rgba(0,0,0,0.28),0 2px 6px rgba(0,0,0,0.18),inset 0 1px 0 rgba(255,255,255,0.6);
    opacity:0; transform:translateY(14px) scale(0.9);
    max-width:260px;
  }}
  .card.left {{ animation:enter 0.5s cubic-bezier(0.22,1,0.36,1) forwards; }}
  .card.right {{ animation:enter 0.5s cubic-bezier(0.22,1,0.36,1) 0.18s forwards; }}
  @keyframes enter {{
    0% {{ opacity:0; transform:translateY(14px) scale(0.9); }}
    70% {{ opacity:1; transform:translateY(-3px) scale(1.02); }}
    100% {{ opacity:1; transform:translateY(0) scale(1); }}
  }}
  .icon-wrap {{ width:38px;height:38px;border-radius:10px; display:flex;align-items:center;justify-content:center; }}
  .icon-wrap svg {{ width:20px;height:20px; }}
  .label {{ font-size:22px; font-weight:650; letter-spacing:-0.2px; color:{text_color}; line-height:1.15; }}
  .vs {{ font-size:16px; font-weight:700; color:{text_color}; opacity:0.6; }}
</style></head><body>
  <div class="wrap">
    <div class="card left">
      <div class="icon-wrap" style="background:linear-gradient(135deg,#6a8dff,#4fd1ff)">{left_icon}</div>
      <div class="label">{left_text}</div>
    </div>
    <div class="vs">VS</div>
    <div class="card right">
      <div class="icon-wrap" style="background:linear-gradient(135deg,#ff9a6a,#ff5c8a)">{right_icon}</div>
      <div class="label">{right_text}</div>
    </div>
  </div>
</body></html>
"""


def render_animated_comparison_frames(left_text: str, right_text: str, left_icon: str, right_icon: str,
                                       fps: int, duration_s: float, light_theme: bool = False,
                                       out_dir: str = "anim_frames_cmp", fade_out_s: float = 0.25):
    """Public entry point — see _render_animated_comparison_frames_sync()
    for the actual implementation. Runs it in an isolated thread, same
    reasoning as render_animated_label_frames()."""
    return _run_isolated(_render_animated_comparison_frames_sync, left_text, right_text, left_icon,
                          right_icon, fps, duration_s, light_theme, out_dir, fade_out_s)


def _render_animated_comparison_frames_sync(left_text: str, right_text: str, left_icon: str, right_icon: str,
                                             fps: int, duration_s: float, light_theme: bool = False,
                                             out_dir: str = "anim_frames_cmp", fade_out_s: float = 0.25):
    """Same entrance/hold/fade-out lifecycle as render_animated_label_frames,
    but for the two-card comparison layout — right card staggers in
    0.18s after the left one instead of both popping in together."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    bg_alpha, bg_alpha2, text_color = ("0.55", "0.28", "#16182a") if light_theme else ("0.18", "0.10", "#f2f4ff")

    html = _COMPARISON_TEMPLATE.format(
        bg_alpha=bg_alpha, bg_alpha2=bg_alpha2, text_color=text_color,
        left_icon=_ICONS.get(left_icon, _ICONS["check"]), right_icon=_ICONS.get(right_icon, _ICONS["check"]),
        left_text=_wrap_label(left_text), right_text=_wrap_label(right_text),
    )
    html_path = f"{out_dir}/_card.html"
    with open(html_path, "w") as f:
        f.write(html)

    n_frames = max(1, int(fps * duration_s))
    frame_paths = []
    PAD = 24
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 900, "height": 220})
        page.goto(f"file://{os.path.abspath(html_path)}")
        wrap = page.query_selector(".wrap")

        page.evaluate(
            """(t) => { document.getAnimations().forEach(a => { a.pause(); a.currentTime = t; }); }""",
            (duration_s + 1) * 1000,
        )
        settled = wrap.bounding_box()
        clip_rect = {
            "x": max(0, settled["x"] - PAD), "y": max(0, settled["y"] - PAD),
            "width": settled["width"] + 2 * PAD, "height": settled["height"] + 2 * PAD,
        }
        box = (int(clip_rect["width"]), int(clip_rect["height"]))

        for i in range(n_frames):
            t = i / fps
            page.evaluate(
                """(t) => { document.getAnimations().forEach(a => { a.pause(); a.currentTime = t; }); }""",
                t * 1000,
            )
            path = f"{out_dir}/f{i:04d}.png"
            page.screenshot(path=path, clip=clip_rect, omit_background=True)
            _apply_fade_tail(path, t, duration_s, fade_out_s)
            frame_paths.append(path)
        browser.close()

    return frame_paths, box[0], box[1]


def build_moviepy_layer(frame_paths, fps, start, target_x_frac, target_y_frac, vid_w, vid_h):
    """Turns a rendered PNG sequence into a positioned, transparency-aware
    MoviePy clip.

    IMPORTANT: pass frame_paths (file path STRINGS) straight to
    ImageSequenceClip, don't pre-load them into numpy arrays first.
    MoviePy 1.0.3's ImageSequenceClip has two different code paths
    depending on whether `sequence` is a list of file paths or a list
    of already-loaded numpy arrays. The file-path path correctly
    extracts a real per-frame alpha mask from RGBA PNGs via
    `with_mask=True` (imread's 4th channel). The numpy-array path,
    however, unconditionally does `sequence[index][:,:,:3]` in its
    make_frame — regardless of the with_mask/ismask flags — which
    crashes on a 2D mask array with "too many indices for array". An
    earlier version of this function pre-loaded frames into RGB +
    alpha numpy arrays and built two separate ImageSequenceClips to
    combine manually; that hit exactly this crash and was replaced
    with this simpler, correct version, verified against the actual
    installed moviepy==1.0.3 behavior.
    """
    from moviepy.editor import ImageSequenceClip
    from PIL import Image

    with Image.open(frame_paths[0]) as im:
        w, h = im.size

    clip = ImageSequenceClip(frame_paths, fps=fps, with_mask=True)

    x = int(vid_w * target_x_frac - w / 2)
    y = int(vid_h * target_y_frac)
    return clip.set_start(start).set_position((x, y))