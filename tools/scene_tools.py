import os
import re
import math
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SCENE_HOOK      = "HOOK"
SCENE_FACT      = "FACT"
SCENE_STATISTIC = "STATISTIC"
SCENE_QUOTE     = "QUOTE"
SCENE_QUESTION  = "QUESTION"
SCENE_REVEAL    = "REVEAL"
SCENE_EVIDENCE  = "EVIDENCE"
SCENE_CTA       = "CTA"

ALL_SCENE_TYPES = [
    SCENE_HOOK, SCENE_FACT, SCENE_STATISTIC, SCENE_QUOTE,
    SCENE_QUESTION, SCENE_REVEAL, SCENE_EVIDENCE, SCENE_CTA,
]

SCENE_CAMERA = {
    SCENE_HOOK:      "push_forward",
    SCENE_FACT:      "pan_left",
    SCENE_STATISTIC: "zoom_in",
    SCENE_QUOTE:     "pull_back",
    SCENE_QUESTION:  "pan_right",
    SCENE_REVEAL:    "zoom_in",
    SCENE_EVIDENCE:  "pan_left",
    SCENE_CTA:       "zoom_out",
}

_KW_PATTERNS = [
    (r'\$[\d,.]+[BMKbmk]?\b',                                  "money"),
    (r'[\d,.]+\s*%',                                            "percent"),
    (r'[\d,.]+\s*(?:million|billion|trillion)\b',              "number"),
    (r'\b\d{4}\b',                                              "date"),
    (r'\b\d+[xX]\b',                                           "multiplier"),
    (r'\b\d{1,3}(?:,\d{3})+\b',                               "number"),
    (r'\b(?:January|February|March|April|May|June|July'
     r'|August|September|October|November|December)\s+\d{4}', "date"),
]

KEYWORD_COLORS = {
    "money":      (255, 215,  0),
    "percent":    ( 80, 220, 180),
    "number":     (255, 200,  50),
    "date":       (180, 180, 255),
    "multiplier": (255, 160,  80),
}

_QUESTION_ENDINGS  = ["?"]
_QUOTE_TRIGGERS    = ["according to", " said ", " stated ", " claims ", " told "]
_REVEAL_TRIGGERS   = ["revealed", "discovered", "turns out", "actually", "but wait"]
_EVIDENCE_TRIGGERS = ["study", "research", "data shows", "scientists", "experts"]


def classify_scene(text: str, idx: int, total: int) -> str:
    t   = text.strip()
    low = t.lower()

    if idx == 0:
        return SCENE_HOOK
    if idx >= total - 1:
        return SCENE_CTA
    if t.endswith("?"):
        return SCENE_QUESTION
    if t.startswith(('"', "“")) or any(tr in low for tr in _QUOTE_TRIGGERS):
        return SCENE_QUOTE
    if any(tr in low for tr in _EVIDENCE_TRIGGERS):
        return SCENE_EVIDENCE
    if any(re.search(p, t, re.IGNORECASE) for p, _ in _KW_PATTERNS):
        return SCENE_STATISTIC
    if any(tr in low for tr in _REVEAL_TRIGGERS) or idx == total // 2:
        return SCENE_REVEAL
    return SCENE_FACT


def detect_keywords(text: str) -> list[tuple]:
    spans = []
    for pattern, ktype in _KW_PATTERNS:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            color = KEYWORD_COLORS.get(ktype, (255, 255, 255))
            spans.append((m.start(), m.end(), ktype, color))
    spans.sort(key=lambda s: s[0])
    deduped, last_end = [], -1
    for s in spans:
        if s[0] >= last_end:
            deduped.append(s)
            last_end = s[1]
    return deduped


_NUM_PAT = re.compile(
    r'(\$?)([\d,.]+)'
    r'(\s*[BMK%]\b|\s*x\b|\s+(?:million|billion|trillion)\b)?',
    re.IGNORECASE,
)


def extract_number_info(text: str) -> tuple:
    m = _NUM_PAT.search(text)
    if not m:
        return None, "", ""
    prefix  = m.group(1)
    raw     = m.group(2).replace(",", "")
    suffix  = (m.group(3) or "").strip()
    try:
        val = float(raw)
    except Exception:
        return None, "", ""
    return val, prefix, suffix


def count_up_text(text: str, t: float, seg_dur: float) -> str:
    val, prefix, suffix = extract_number_info(text)
    if val is None:
        return text
    if not prefix and not suffix and 1800 <= val <= 2199:
        return text
    progress = min(1.0, t / max(seg_dur * 0.72, 0.1))
    current  = val * progress
    if val >= 1_000_000_000:
        disp = f"{current / 1_000_000_000:.1f}B"
    elif val >= 1_000_000:
        disp = f"{current / 1_000_000:.1f}M"
    elif val >= 1_000:
        disp = f"{current:,.0f}"
    elif val >= 10:
        disp = f"{current:.0f}"
    else:
        disp = f"{current:.1f}"
    sep     = " " if suffix and len(suffix) > 1 else ""
    num_str = f"{prefix}{disp}{sep}{suffix}" if suffix else f"{prefix}{disp}"
    m = _NUM_PAT.search(text)
    if m:
        return text[:m.start()] + num_str + text[m.end():]
    return text


_SCRAMBLE_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789#@!?"

def scramble_text(text: str, t: float, dur: float, seed: int = 0) -> str:
    rng      = random.Random(seed + int(t * 20))
    p        = min(1.0, t / max(dur, 0.01))
    revealed = int(len(text) * p)
    return "".join(
        ch if ch == " " or i < revealed else rng.choice(_SCRAMBLE_CHARS)
        for i, ch in enumerate(text)
    )


def estimate_word_timing(text: str, t_start: float, t_end: float) -> list[tuple]:
    words = text.split()
    if not words:
        return []
    dur  = max(0.05, t_end - t_start)
    wdur = dur / len(words)
    return [
        (w, t_start + i * wdur, t_start + (i + 1) * wdur)
        for i, w in enumerate(words)
    ]


def render_word_captions(
    frame:        np.ndarray,
    words_timing: list[tuple],
    abs_t:        float,
    caption_y:    int | None = None,
    font_size:    int        = 50,
    style:        int        = 0,
) -> np.ndarray:
    if not words_timing:
        return frame

    H, W = frame.shape[:2]
    img  = Image.fromarray(frame).convert("RGBA")

    cur_idx = 0
    for i, (_, ts, _) in enumerate(words_timing):
        if ts <= abs_t:
            cur_idx = i

    WINDOW = 8
    wstart = max(0, cur_idx - 2)
    wend   = min(len(words_timing), wstart + WINDOW)
    wstart = max(0, wend - WINDOW)
    visible = words_timing[wstart:wend]

    overlay     = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw        = ImageDraw.Draw(overlay)
    font_active = _bold(font_size)
    font_other  = _bold(int(font_size * 0.78))

    if caption_y is None:
        caption_y = H - 200

    parts = []
    total_w = 0
    for j, (word, _, _) in enumerate(visible):
        is_cur  = (wstart + j == cur_idx)
        display = word.upper() if (style == 2 and is_cur) else word
        f       = font_active if is_cur else font_other
        bw      = _text_w(f, display)
        parts.append((display, is_cur, f, bw))
        total_w += bw + 16

    bar_h = font_size + 52
    draw.rectangle([(0, caption_y - 16), (W, caption_y + bar_h)], fill=(0, 0, 0, 148))

    x = max(28, (W - total_w) // 2)
    for display, is_cur, f, bw in parts:
        bh = _text_h(f, display)
        y  = caption_y + 10

        if is_cur:
            if style == 0:
                draw.rectangle([(x - 7, y - 5), (x + bw + 7, y + bh + 5)], fill=(255, 200, 50, 225))
                draw.text((x, y), display, fill=(8, 8, 8, 255), font=f)
            else:
                draw.text((x, y), display, fill=(255, 250, 235, 255), font=f)
        else:
            draw.text((x, y), display, fill=(130, 130, 130, 200), font=f)

        x += bw + 16

    img = Image.alpha_composite(img, overlay)
    return np.array(img.convert("RGB"))


def draw_rich_text(
    draw:    ImageDraw.Draw,
    text:    str,
    x:       int,
    y:       int,
    font,
    default: tuple,
    kws:     list[tuple],
) -> int:
    if not kws:
        draw.text((x, y), text, fill=(*default[:3], 255), font=font)
        return x + _text_w(font, text)

    pos, cx = 0, x
    for ks, ke, _, kcolor in sorted(kws, key=lambda s: s[0]):
        if pos < ks:
            chunk = text[pos:ks]
            draw.text((cx, y), chunk, fill=(*default[:3], 255), font=font)
            cx += _text_w(font, chunk)
        ktext = text[ks:ke]
        draw.text((cx, y), ktext, fill=(*kcolor, 255), font=font)
        cx += _text_w(font, ktext)
        pos = ke
    if pos < len(text):
        rest = text[pos:]
        draw.text((cx, y), rest, fill=(*default[:3], 255), font=font)
        cx += _text_w(font, rest)
    return cx


def apply_camera_move(
    frame:   np.ndarray,
    move:    str,
    t:       float,
    seg_dur: float,
) -> np.ndarray:
    from tools.video_tools import _zoom_frame_pan
    p = min(1.0, t / max(seg_dur, 0.01))

    if move == "push_forward":
        return _zoom_frame_pan(frame, 1.0 + 0.14 * p, 0.5)
    if move == "pull_back":
        return _zoom_frame_pan(frame, 1.14 - 0.14 * p, 0.5)
    if move == "pan_left":
        return _zoom_frame_pan(frame, 1.07, 0.53 - 0.05 * p)
    if move == "pan_right":
        return _zoom_frame_pan(frame, 1.07, 0.47 + 0.05 * p)
    if move == "zoom_in":
        return _zoom_frame_pan(frame, 1.0 + 0.10 * p, 0.5)
    if move == "zoom_out":
        return _zoom_frame_pan(frame, 1.10 - 0.10 * p, 0.5)
    return frame


def render_hierarchy_text(
    draw:       ImageDraw.Draw,
    text:       str,
    x:          int,
    y:          int,
    level:      int,
    accent:     tuple = (255, 200, 50),
    max_width:  int   = 980,
    line_h:     int   = 70,
) -> int:
    sizes    = {1: 84, 2: 56, 3: 40}
    colors   = {1: (255, 250, 235), 2: (220, 215, 200), 3: accent}
    font_sz  = sizes.get(level, 56)
    color    = colors.get(level, (255, 255, 255))
    font     = _bold(font_sz)
    line_h   = {1: 95, 2: 65, 3: 48}.get(level, 65)

    from tools.video_tools import _wrap_text_str
    lines = _wrap_text_str(text, font, max_width)
    for line in lines[:4]:
        draw.text((x, y), line, fill=(*color, 255), font=font)
        y += line_h
    return y


def render_scene_overlay(
    frame:      np.ndarray,
    scene_type: str,
    text:       str,
    t:          float,
    seg_dur:    float,
    accent:     tuple = (255, 200, 50),
    rank:       int   = 0,
) -> np.ndarray:
    H, W    = frame.shape[:2]
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)
    pad     = 52

    if scene_type == SCENE_STATISTIC:
        _render_statistic(draw, text, t, seg_dur, W, H, pad, accent)
    elif scene_type == SCENE_QUOTE:
        _render_quote(draw, text, t, seg_dur, W, H, pad, accent)
    elif scene_type == SCENE_QUESTION:
        _render_question(draw, text, t, seg_dur, W, H, pad, accent)
    elif scene_type == SCENE_REVEAL:
        _render_reveal(draw, text, t, seg_dur, W, H, pad, accent)
    elif scene_type == SCENE_HOOK:
        _render_hook_text(draw, text, t, seg_dur, W, H, pad, accent)
    else:
        _render_fact(draw, text, t, seg_dur, W, H, pad, accent)

    img = Image.fromarray(frame).convert("RGBA")
    img = Image.alpha_composite(img, overlay)
    return np.array(img.convert("RGB"))


def _render_statistic(draw, text, t, seg_dur, W, H, pad, accent):
    animated = count_up_text(text, t, seg_dur)
    kws      = detect_keywords(animated)
    font     = _bold(92)
    lines    = _wrap_lines(animated, font, W - pad * 2)
    char_off = 0
    y = H // 2 - len(lines) * 105 // 2
    for line in lines[:3]:
        line_kws = _adjust_kws(kws, char_off, line)
        draw_rich_text(draw, line, pad, y, font, (255, 250, 235), line_kws)
        y += 105
        char_off += len(line) + 1


def _render_quote(draw, text, t, seg_dur, W, H, pad, accent):
    chars   = max(1, int(len(text) * min(1.0, t / max(seg_dur * 0.8, 0.1))))
    cursor  = "▌" if t / max(seg_dur, 0.01) < 0.9 and int(t * 4) % 2 == 0 else ""
    display = text[:chars] + cursor
    font    = _bold(62)
    draw.text((pad, H // 2 - 160), "“", fill=(*accent, 200), font=_bold(120))
    lines = _wrap_lines(display, font, W - pad * 2 - 40)
    y = H // 2 - 60
    for line in lines[:4]:
        draw.text((pad + 36, y), line, fill=(255, 250, 235, 255), font=font)
        y += 74


def _render_question(draw, text, t, seg_dur, W, H, pad, accent):
    pulse  = 0.92 + 0.08 * math.sin(t * math.pi * 2.5)
    alpha  = int(255 * min(1.0, t / 0.20) * pulse)
    font   = _bold(78)
    lines  = _wrap_lines(text, font, W - pad * 2)
    total_h = len(lines) * 90
    y = H // 2 - total_h // 2
    for line in lines[:4]:
        draw.text((pad, y), line, fill=(255, 250, 235, alpha), font=font)
        y += 90
    glow_a = int(180 * min(1.0, (t - 0.3) / 0.4)) if t > 0.3 else 0
    draw.rectangle([(pad, y + 8), (W - pad, y + 12)], fill=(*accent, glow_a))


def _render_reveal(draw, text, t, seg_dur, W, H, pad, accent):
    dur     = min(seg_dur * 0.6, 1.2)
    display = scramble_text(text, t, dur, seed=hash(text) % 1000)
    font    = _bold(76)
    lines   = _wrap_lines(display, font, W - pad * 2)
    y = H // 2 - len(lines) * 88 // 2
    for line in lines[:4]:
        draw.text((pad, y), line, fill=(255, 250, 235, 255), font=font)
        y += 88
    if t > dur:
        arr_a = int(220 * min(1.0, (t - dur) / 0.3))
        draw.text((W // 2 - 16, y + 12), "▼", fill=(*accent, arr_a), font=_bold(48))


def _render_hook_text(draw, text, t, seg_dur, W, H, pad, accent):
    alpha = int(255 * min(1.0, t / 0.25))
    font  = _bold(96)
    lines = _wrap_lines(text.upper(), font, W - pad * 2)
    total_h = len(lines) * 110
    y = H // 2 - total_h // 2
    for line in lines[:3]:
        draw.text((pad, y), line, fill=(255, 250, 235, alpha), font=font)
        y += 110


def _render_fact(draw, text, t, seg_dur, W, H, pad, accent):
    if t < 0.14:
        offset = int(55 * (1 - t / 0.14) ** 2)
    elif t < 0.24:
        offset = int(-8 * math.sin(math.pi * (t - 0.14) / 0.10))
    else:
        offset = 0
    alpha = int(255 * min(1.0, t / 0.12))
    kws   = detect_keywords(text)
    font  = _bold(66)
    lines = _wrap_lines(text, font, W - pad * 2)
    y = H // 2 - len(lines) * 76 // 2 + offset
    char_off = 0
    for line in lines[:4]:
        line_kws = _adjust_kws(kws, char_off, line)
        if line_kws:
            draw_rich_text(draw, line, pad, y, font, (255, 250, 235), line_kws)
        else:
            draw.text((pad, y), line, fill=(255, 250, 235, alpha), font=font)
        y += 76
        char_off += len(line) + 1


def _adjust_kws(kws, char_offset, line):
    line_end = char_offset + len(line)
    result   = []
    for ks, ke, kt, kc in kws:
        if ks >= char_offset and ke <= line_end:
            result.append((ks - char_offset, ke - char_offset, kt, kc))
    return result


def _wrap_lines(text, font, max_width):
    words, lines, current = text.split(), [], ""
    for word in words:
        test = (current + " " + word).strip()
        if _text_w(font, test) > max_width and current:
            lines.append(current)
            current = word
        else:
            current = test
    if current:
        lines.append(current)
    return lines or [""]


_FONT_CACHE_LOCAL: dict = {}

def _bold(size: int):
    key = f"bold_{size}"
    if key not in _FONT_CACHE_LOCAL:
        for path, idx in [
            ("/System/Library/Fonts/HelveticaNeue.ttc", 1),
            ("/System/Library/Fonts/ArialHB.ttc",       0),
            ("/System/Library/Fonts/Helvetica.ttc",     1),
            ("/System/Library/Fonts/Arial.ttf",         0),
            ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 0),
        ]:
            if os.path.exists(path):
                try:
                    _FONT_CACHE_LOCAL[key] = ImageFont.truetype(path, size, index=idx)
                    break
                except Exception:
                    continue
        if key not in _FONT_CACHE_LOCAL:
            _FONT_CACHE_LOCAL[key] = ImageFont.load_default()
    return _FONT_CACHE_LOCAL[key]


def _text_w(font, text: str) -> int:
    try:
        return font.getbbox(text)[2] - font.getbbox(text)[0]
    except Exception:
        return len(text) * 24


def _text_h(font, text: str) -> int:
    try:
        return font.getbbox(text)[3] - font.getbbox(text)[1]
    except Exception:
        return 40
