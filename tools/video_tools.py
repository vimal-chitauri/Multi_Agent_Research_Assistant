import os
import math
import time
import requests
import textwrap
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv

load_dotenv(override=True)

VIDEOS_DIR = Path(__file__).parent.parent / "generated_videos"
MUSIC_DIR  = Path(__file__).parent.parent / "generated_videos" / "music_cache"
VIDEOS_DIR.mkdir(exist_ok=True)
MUSIC_DIR.mkdir(exist_ok=True)

VID_W = 1080
VID_H = 1920

ANIMATION_EFFECTS = ["slide_up", "fade", "slide_left", "typewriter"]

_HOOK_TEXTS = {
    "technology": "YOU NEED\nTO KNOW THIS",
    "fitness":    "THIS WILL\nCHANGE EVERYTHING",
    "motivation": "THIS WILL\nHIT DIFFERENT",
    "finance":    "MONEY MOVES\nTHAT MATTER",
    "crypto":     "BEFORE YOU\nBUY ANYTHING",
    "marketing":  "WHAT TOP\nBRANDS DO",
    "food":       "YOU'VE BEEN\nDOING IT WRONG",
    "travel":     "HIDDEN SPOTS\nYOU MUST SEE",
    "latest_news": "JUST IN ↓",
}
_NEWS_HOOK_TEXTS = [
    "YOU NEED TO\nSEE THIS",
    "THIS JUST\nHAPPENED",
    "WAIT FOR\nTHIS ↓",
    "EVERYONE IS\nTALKING ABOUT THIS",
]

NICHE_MUSIC_QUERIES = {
    "latest_news": "news music background",
    "fitness":    "energetic workout music background",
    "motivation": "inspirational uplifting background music",
    "technology": "electronic ambient instrumental",
    "finance":    "corporate professional background music",
    "crypto":     "digital electronic music",
    "marketing":  "upbeat pop background music",
    "food":       "acoustic cheerful background music",
    "travel":     "adventure ambient world music",
}

NICHE_VIDEO_PROMPTS = {
    "latest_news": "Modern news studio, breaking news broadcast, dynamic graphics, professional lighting, {topic}, cinematic 4K",
    "fitness":    "Person exercising outdoors, energetic workout, dynamic motion, sunlight, {topic}, cinematic",
    "motivation": "Inspiring sunrise landscape, person achieving goals, silhouette, {topic}, cinematic 4K",
    "technology": "Futuristic digital interface, glowing data streams, blue light, modern tech, {topic}, cinematic",
    "finance":    "Stock market graphs rising, professional city skyline, {topic}, cinematic",
    "crypto":     "Abstract blockchain network, glowing particles, digital currency, {topic}, dark cinematic",
    "marketing":  "Creative workspace, vibrant colors, content creation, {topic}, modern cinematic",
    "food":       "Fresh ingredients, warm kitchen light, cooking, {topic}, cinematic 4K",
    "travel":     "Aerial drone shot, beautiful destination, golden hour, {topic}, cinematic 4K",
}

SHOT_DEFS = [
    (0.50, 0.50, 1.00),
    (0.42, 0.50, 1.10),
    (0.58, 0.50, 1.10),
    (0.50, 0.40, 1.10),
    (0.50, 0.60, 1.10),
    (0.44, 0.44, 1.15),
    (0.56, 0.56, 1.15),
    (0.44, 0.56, 1.12),
]

_FONT_CACHE: dict[int, ImageFont.FreeTypeFont] = {}
_GRADIENT_CACHE: dict[str, np.ndarray] = {}


def group_bullet_points(info_lines: list[str]) -> list[list[str]]:
    SHORT_MAX = 55
    cleaned   = [l.strip().lstrip("•-–▶ ") for l in info_lines if l.strip()]
    groups    = []
    i = 0
    while i < len(cleaned):
        cur  = cleaned[i]
        nxt  = cleaned[i + 1] if i + 1 < len(cleaned) else None
        if len(cur) <= SHORT_MAX and nxt and len(nxt) <= SHORT_MAX:
            groups.append([cur, nxt])
            i += 2
        else:
            groups.append([cur])
            i += 1
    return groups[:10] or [["Stay informed and keep learning"]]


def search_freesound_music(niche: str, rank: int = 0) -> dict | None:
    api_key = os.getenv("FREESOUND_API_KEY")
    if not api_key:
        return None
    query = NICHE_MUSIC_QUERIES.get(niche.lower(), "ambient background music")
    print(f"  Music: '{query}' | track #{rank % 10}")
    try:
        r = requests.get(
            "https://freesound.org/apiv2/search/text/",
            params={"query": query, "token": api_key,
                    "fields": "id,name,previews,duration,username",
                    "filter": "duration:[15 TO 120]",
                    "page_size": 15, "sort": "rating_desc"},
            timeout=10,
        )
        results = [t for t in r.json().get("results", []) if t.get("previews", {}).get("preview-hq-mp3")]
        if results:
            track = results[rank % len(results)]
            print(f"  Found: '{track['name']}' by {track.get('username', '?')}")
            return track
    except Exception as e:
        print(f"  Freesound error: {e}")
    return None


def download_music(track: dict, filename: str) -> str | None:
    api_key = os.getenv("FREESOUND_API_KEY", "")
    url = track.get("previews", {}).get("preview-hq-mp3")
    if not url:
        return None
    try:
        path = MUSIC_DIR / filename
        if path.exists():
            return str(path)
        r = requests.get(url, params={"token": api_key}, timeout=30, stream=True)
        if r.status_code == 200:
            with open(path, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
            return str(path)
    except Exception as e:
        print(f"  Music download error: {e}")
    return None



def _create_shot_from_image(img_arr: np.ndarray, shot_idx: int) -> np.ndarray:
    h, w         = img_arr.shape[:2]
    cx, cy, zoom = SHOT_DEFS[shot_idx % len(SHOT_DEFS)]
    cw = int(w / zoom)
    ch = int(h / zoom)
    x1 = max(0, min(int(cx * w - cw // 2), w - cw))
    y1 = max(0, min(int(cy * h - ch // 2), h - ch))
    crop = img_arr[y1:y1+ch, x1:x1+cw]
    return np.array(Image.fromarray(crop).resize((VID_W, VID_H), Image.LANCZOS))


def _create_shots(img_arr: np.ndarray, n: int) -> list[np.ndarray]:
    return [_create_shot_from_image(img_arr, i) for i in range(n)]


def _zoom_frame(arr: np.ndarray, zoom: float) -> np.ndarray:
    h, w = arr.shape[:2]
    ch   = int(h / zoom)
    cw   = int(w / zoom)
    y1   = (h - ch) // 2
    x1   = (w - cw) // 2
    return np.array(Image.fromarray(arr[y1:y1+ch, x1:x1+cw]).resize((w, h), Image.LANCZOS))


def _zoom_frame_pan(arr: np.ndarray, zoom: float, cx: float = 0.5) -> np.ndarray:
    h, w = arr.shape[:2]
    ch   = int(h / zoom)
    cw   = int(w / zoom)
    x_c  = int(cx * w)
    y1   = (h - ch) // 2
    x1   = max(0, min(x_c - cw // 2, w - cw))
    return np.array(Image.fromarray(arr[y1:y1+ch, x1:x1+cw]).resize((w, h), Image.LANCZOS))


def _punch_scale_and_alpha(t: float, dur: float = 0.35) -> tuple[float, float]:
    p = min(1.0, t / dur)
    if p < 0.40:
        scale = 0.6 + 1.45 * p
    elif p < 0.70:
        scale = 1.18 - 0.60 * (p - 0.40)
    else:
        scale = 1.0
    alpha = min(1.0, p / 0.45)
    return scale, alpha


def _apply_punch_scale(frame: np.ndarray, scale: float) -> np.ndarray:
    if abs(scale - 1.0) < 0.004:
        return frame
    H, W = frame.shape[:2]
    if scale > 1.0:
        new_h = max(1, int(H / scale))
        new_w = max(1, int(W / scale))
        y1 = (H - new_h) // 2
        x1 = (W - new_w) // 2
        return np.array(Image.fromarray(frame[y1:y1+new_h, x1:x1+new_w]).resize((W, H), Image.LANCZOS))
    else:
        new_h = int(H * scale)
        new_w = int(W * scale)
        small = np.array(Image.fromarray(frame).resize((new_w, new_h), Image.LANCZOS))
        out   = np.zeros((H, W, 3), dtype=np.uint8)
        out[(H - new_h)//2:(H - new_h)//2 + new_h, (W - new_w)//2:(W - new_w)//2 + new_w] = small
        return out


def weighted_segment_durations(groups: list, total_duration: float, min_secs: float = 3.0) -> list[float]:
    counts = [max(1, sum(len(p.split()) for p in g)) for g in groups]
    total  = sum(counts)
    raw    = [total_duration * c / total for c in counts]
    durations = [max(min_secs, d) for d in raw]
    excess = sum(durations) - total_duration
    if excess > 0:
        adj = [d - min_secs for d in durations]
        total_adj = sum(adj) or 1
        durations = [min_secs + a - excess * a / total_adj for a in adj]
    return durations


def _render_hook_card(frame_bg: np.ndarray, text: str, t: float) -> np.ndarray:
    H, W = frame_bg.shape[:2]
    img = Image.fromarray(frame_bg)
    img = img.resize((W // 18, H // 18), Image.BILINEAR).resize((W, H), Image.NEAREST).convert("RGBA")
    dark = Image.new("RGBA", (W, H), (0, 0, 0, 210))
    img  = Image.alpha_composite(img, dark)

    scale, alpha_f = _punch_scale_and_alpha(t, dur=0.35)
    font  = _load_bold_font(100)
    lines = text.split("\n")
    WARM  = (255, 250, 235)

    txt_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    td = ImageDraw.Draw(txt_layer)
    total_h = len(lines) * 116
    sy = (H - total_h) // 2
    for line in lines:
        try:
            bw = font.getbbox(line)[2] - font.getbbox(line)[0]
        except Exception:
            bw = len(line) * 50
        td.text(((W - bw) // 2, sy), line, fill=(*WARM, int(255 * alpha_f)), font=font)
        sy += 116

    if abs(scale - 1.0) > 0.01:
        new_w = max(1, int(W * scale))
        new_h = max(1, int(H * scale))
        txt_layer = txt_layer.resize((new_w, new_h), Image.LANCZOS)
        x_off = (new_w - W) // 2
        y_off = (new_h - H) // 2
        txt_layer = txt_layer.crop((x_off, y_off, x_off + W, y_off + H))

    if t > 0.5:
        ad    = ImageDraw.Draw(txt_layer)
        arr_a = int(200 * min(1.0, (t - 0.5) / 0.4))
        af    = _load_bold_font(56)
        ad.text((W // 2 - 16, H - 190), "▼", fill=(255, 200, 50, arr_a), font=af)

    img = Image.alpha_composite(img, txt_layer)
    return np.array(img.convert("RGB"))


def _generate_base_frames(
    shots: list[np.ndarray],
    duration: int,
    fps: int,
    trans_secs: float = 0.6,
) -> list[np.ndarray]:
    n      = len(shots)
    total  = duration * fps
    FLASH  = max(2, int(fps * 0.083))
    base   = total // n
    lens   = [base] * n
    lens[-1] += total - sum(lens)

    all_frames: list[np.ndarray] = []

    for seg_idx, (shot, slen) in enumerate(zip(shots, lens)):
        zoom_in  = (seg_idx % 2 == 0)
        pan_sign = 1 if (seg_idx // 2) % 2 == 0 else -1

        for fi in range(slen):
            p     = fi / max(slen - 1, 1)
            zoom  = 1.0 + 0.10 * p if zoom_in else 1.10 - 0.10 * p
            cx    = 0.5 + pan_sign * 0.025 * p
            frame = _zoom_frame_pan(shot, zoom, cx)

            if seg_idx < n - 1:
                frames_left = slen - fi
                if frames_left <= FLASH:
                    flash_t = 1.0 - frames_left / FLASH
                    frame = (
                        frame.astype(np.float32) * (1 - flash_t * 0.65)
                        + 255 * flash_t * 0.65
                    ).clip(0, 255).astype(np.uint8)

            all_frames.append(frame)

    return all_frames[:total]


def _get_gradient_base(niche: str) -> tuple[np.ndarray, int, tuple]:
    if niche not in _GRADIENT_CACHE:
        W, H        = VID_W, VID_H
        overlay_top = int(H * 0.60)
        accent      = _niche_accent_color(niche)

        img  = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        for y in range(overlay_top, H):
            p = (y - overlay_top) / (H - overlay_top)
            a = int(225 * p + 15)
            draw.rectangle([(0, y), (W, y + 1)], fill=(0, 0, 0, min(a, 242)))

        draw.rectangle([(0, overlay_top), (W, overlay_top + 3)], fill=(*accent, 210))

        _GRADIENT_CACHE[niche] = (np.array(img), overlay_top, accent)

    return _GRADIENT_CACHE[niche]


def _render_segment_overlay(
    seg_t: float,
    seg_dur: float,
    title: str,
    points: list[str],
    niche: str,
    seg_idx: int,
    n_groups: int,
    effect: str,
    trans_secs: float = 0.6,
    watermark: str = "",
) -> np.ndarray:
    gradient_arr, overlay_top, accent = _get_gradient_base(niche)

    img  = Image.fromarray(gradient_arr.copy())
    draw = ImageDraw.Draw(img)
    W, H = VID_W, VID_H
    pad  = 52

    ENTER_DUR = 0.15
    in_p      = min(seg_t / ENTER_DUR, 1.0)

    EXIT_DUR = trans_secs
    is_last  = (seg_idx >= n_groups - 1)
    out_p    = 0.0
    if not is_last and seg_dur > EXIT_DUR:
        raw   = (seg_t - (seg_dur - EXIT_DUR)) / EXIT_DUR
        out_p = max(0.0, min(raw, 1.0))

    f_title  = _load_font(40)
    f_bullet = _load_font(35)
    f_small  = _load_font(21)

    title_a = int(255 * (min(seg_t / 0.4, 1.0) if seg_idx == 0 else 1.0))
    ttxt    = title[:50] + ("..." if len(title) > 50 else "")
    draw.text((pad, overlay_top + 12), ttxt, fill=(*accent, title_a), font=f_title)

    div_a = int(90 * (min(seg_t / 0.4, 1.0) if seg_idx == 0 else 1.0))
    draw.rectangle([(pad, overlay_top + 56), (W - pad, overlay_top + 58)],
                   fill=(255, 255, 255, div_a))

    LINE_H  = 56
    WRAP_W  = 36
    start_y = overlay_top + 70

    for p_idx, point in enumerate(points[:2]):
        base_y  = start_y + p_idx * LINE_H
        wrapped = textwrap.fill(point.strip(), width=WRAP_W)
        wlines  = wrapped.split("\n")[:2]

        if effect == "fade":
            a = int(255 * (1.0 - out_p))
            for j, wl in enumerate(wlines):
                pfx = "•  " if j == 0 else "    "
                draw.text((pad, base_y + j * 42), pfx + wl, fill=(235, 235, 235, a), font=f_bullet)

        elif effect == "slide_up":
            if seg_t < 0.12:
                t_n    = seg_t / 0.12
                offset = int(55 * (1 - t_n) ** 2)
            elif seg_t < 0.22:
                t_n    = (seg_t - 0.12) / 0.10
                offset = int(-10 * math.sin(math.pi * t_n))
            else:
                offset = 0
            a = int(255 * (1.0 - out_p))
            for j, wl in enumerate(wlines):
                pfx = "•  " if j == 0 else "    "
                draw.text((pad, base_y + j * 42 + offset), pfx + wl, fill=(235, 235, 235, a), font=f_bullet)

        elif effect == "slide_left":
            if seg_t < 0.15:
                t_n    = seg_t / 0.15
                offset = int(220 * (1 - t_n) ** 2)
            else:
                offset = 0
            a = int(255 * (1.0 - out_p))
            for j, wl in enumerate(wlines):
                pfx = "•  " if j == 0 else "    "
                draw.text((pad + offset, base_y + j * 42), pfx + wl, fill=(235, 235, 235, a), font=f_bullet)

        elif effect == "typewriter":
            chars   = max(1, int(len(point) * in_p))
            cursor  = "▌" if in_p < 1.0 and int(seg_t * 3) % 2 == 0 else ""
            display = (point[:chars] + cursor).strip()
            a       = int(255 * (1.0 - out_p))
            draw.text((pad, base_y), "•  " + display, fill=(235, 235, 235, a), font=f_bullet)

    dots = "  ".join("●" if i <= seg_idx else "○" for i in range(n_groups))
    d_a  = int(200 * min(seg_t / 0.5, 1.0))
    d_w  = len(dots) * 13
    draw.text((W - d_w - pad, H - 36), dots, fill=(*accent, d_a), font=f_small)

    if watermark:
        b_a = int(140 * min(seg_t / 0.5, 1.0))
        draw.text((pad, H - 36), watermark, fill=(150, 150, 150, b_a), font=f_small)

    return np.array(img)


def _composite_rgba_over_rgb(frame: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    alpha  = overlay[:, :, 3:4].astype(np.float32) / 255.0
    result = frame.astype(np.float32) * (1.0 - alpha) + overlay[:, :, :3].astype(np.float32) * alpha
    return result.astype(np.uint8)


def create_video(
    image_paths: list[str],
    title: str,
    info_lines: list[str],
    music_path: str | None,
    output_path: str,
    duration: int = 30,
    fps: int = 24,
    niche: str = "technology",
    voice_path: str | None = None,
    rank: int = 0,
    hook: str | None = None,
    watermark: str = "",
) -> str:
    from moviepy.editor import ImageSequenceClip, AudioFileClip, concatenate_audioclips

    groups = group_bullet_points(info_lines)
    n      = len(groups)
    total  = duration * fps
    valid  = [p for p in (image_paths or []) if p and os.path.exists(p)]
    print(f"  {n} segments | {len(info_lines)} bullets | {len(valid)} image(s) → Ken Burns 9:16")

    seg_durations = weighted_segment_durations(groups, float(duration))

    # Build one Ken Burns shot per segment, cycling through all available images
    shots = []
    for i in range(n):
        if valid:
            img_arr = np.array(
                Image.open(valid[i % len(valid)]).convert("RGB")
                .resize((VID_W + 120, VID_H + 200), Image.LANCZOS)
            )
        else:
            img_arr = np.zeros((VID_H + 200, VID_W + 120, 3), dtype=np.uint8)
        shots.append(_create_shot_from_image(img_arr, i))

    base_frames = _generate_base_frames(shots, duration, fps)
    render_fps  = fps

    TRANS_SECS   = 0.6
    FLASH_FRAMES = max(2, int(fps * 0.083))

    seg_lens = [round(d * fps) for d in seg_durations]
    drift = total - sum(seg_lens)
    seg_lens[-1] = max(1, seg_lens[-1] + drift)
    seg_starts = [0]
    for sl in seg_lens[:-1]:
        seg_starts.append(seg_starts[-1] + sl)

    hook_text   = hook or _HOOK_TEXTS.get(niche.lower(), "YOU NEED\nTO SEE THIS")
    hook_frames = int(HOOK_DUR * fps)

    print(f"  Rendering text animations...")
    final_frames: list[np.ndarray] = []

    for seg_idx in range(n):
        s_start = seg_starts[seg_idx]
        s_len   = seg_lens[seg_idx]
        seg_dur = s_len / fps
        effect  = ANIMATION_EFFECTS[(rank + seg_idx) % len(ANIMATION_EFFECTS)]
        points  = groups[seg_idx]

        for fi in range(s_len):
            seg_t = fi / fps
            frame = base_frames[min(s_start + fi, len(base_frames) - 1)]

            if seg_idx == 0 and fi < hook_frames:
                frame = _render_hook_card(frame, hook_text, seg_t)
                if fi >= hook_frames - FLASH_FRAMES:
                    flash_t = 1.0 - (hook_frames - fi) / FLASH_FRAMES
                    frame   = (frame.astype(np.float32) * (1 - flash_t * 0.65)
                               + 255 * flash_t * 0.65).clip(0, 255).astype(np.uint8)
                final_frames.append(frame)
                continue

            overlay = _render_segment_overlay(
                seg_t      = seg_t - (HOOK_DUR if seg_idx == 0 else 0),
                seg_dur    = seg_dur,
                title      = title,
                points     = points,
                niche      = niche,
                seg_idx    = seg_idx,
                n_groups   = n,
                effect     = effect,
                trans_secs = TRANS_SECS,
                watermark  = watermark,
            )
            composited = _composite_rgba_over_rgb(frame, overlay)

            if seg_idx < n - 1 and fi >= s_len - FLASH_FRAMES:
                flash_t    = (fi - (s_len - FLASH_FRAMES)) / FLASH_FRAMES
                composited = (composited.astype(np.float32) * (1 - flash_t * 0.65)
                              + 255 * flash_t * 0.65).clip(0, 255).astype(np.uint8)

            final_frames.append(composited)

    clip = ImageSequenceClip(final_frames, fps=render_fps)

    from moviepy.editor import CompositeAudioClip
    audio_tracks = []

    if voice_path and os.path.exists(voice_path):
        print("  Adding voice narration...")
        voice_clip = AudioFileClip(voice_path)
        if voice_clip.duration > duration:
            voice_clip = voice_clip.subclip(0, duration)
        voice_clip = voice_clip.audio_fadein(0.3).audio_fadeout(1.0)
        audio_tracks.append(voice_clip)

    if music_path and os.path.exists(music_path):
        print("  Adding background music...")
        music_clip = AudioFileClip(music_path)
        if music_clip.duration < duration:
            music_clip = concatenate_audioclips(
                [music_clip] * math.ceil(duration / music_clip.duration)
            )
        music_clip = music_clip.subclip(0, duration).audio_fadein(2).audio_fadeout(3)
        vol = 0.30 if audio_tracks else 1.0
        music_clip = music_clip.volumex(vol)
        audio_tracks.append(music_clip)

    if audio_tracks:
        mixed = CompositeAudioClip(audio_tracks)
        clip  = clip.set_audio(mixed)
    else:
        print("  No audio — silent video")

    print(f"  Rendering → {os.path.basename(output_path)}")
    clip.write_videofile(
        output_path,
        fps=render_fps,
        codec="libx264",
        audio_codec="aac",
        temp_audiofile=str(Path(output_path).parent / f"_tmp_{Path(output_path).stem}.m4a"),
        remove_temp=True,
        logger=None,
    )
    return output_path


def create_image_with_text(
    image_path: str,
    title: str,
    info_lines: list[str],
    output_path: str,
    niche: str = "",
    watermark: str = "",
) -> str:
    img     = Image.open(image_path).convert("RGBA").resize((VID_W, VID_H), Image.LANCZOS)
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)
    overlay_top = int(img.size[1] * 0.60)
    for y in range(overlay_top, img.size[1]):
        p     = (y - overlay_top) / (img.size[1] - overlay_top)
        alpha = int(225 * p + 15)
        draw.rectangle([(0, y), (img.size[0], y + 1)], fill=(0, 0, 0, min(alpha, 242)))
    accent   = _niche_accent_color(niche)
    draw.rectangle([(0, overlay_top), (img.size[0], overlay_top + 3)], fill=(*accent, 210))
    comp     = Image.alpha_composite(img, overlay).convert("RGB")
    draw_rgb = ImageDraw.Draw(comp)
    f_title  = _load_font(40)
    f_bullet = _load_font(35)
    f_small  = _load_font(21)
    pad      = 52
    y        = overlay_top + 12
    draw_rgb.text((pad, y), title[:50], fill=accent, font=f_title)
    y += 56
    for line in info_lines[:4]:
        wl = textwrap.fill(line.strip().lstrip("•-–▶ "), width=36)
        draw_rgb.text((pad, y), "•  " + wl, fill=(235, 235, 235), font=f_bullet)
        y += 56
        if y > comp.size[1] - 60:
            break
    if watermark:
        draw_rgb.text((comp.size[0] - 320, comp.size[1] - 38), watermark,
                      fill=(150, 150, 150), font=f_small)
    comp.save(output_path, "JPEG", quality=95)
    return output_path


def _niche_accent_color(niche: str) -> tuple:
    return {
        "fitness":    (0, 230, 118),
        "motivation": (255, 183, 0),
        "technology": (0, 188, 255),
        "finance":    (0, 230, 118),
        "crypto":     (255, 122, 0),
        "marketing":  (255, 64, 129),
        "food":       (255, 160, 0),
        "travel":     (100, 181, 246),
    }.get(niche.lower(), (255, 255, 255))


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    if size not in _FONT_CACHE:
        for path in [
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        ]:
            if os.path.exists(path):
                try:
                    _FONT_CACHE[size] = ImageFont.truetype(path, size)
                    break
                except Exception:
                    continue
        if size not in _FONT_CACHE:
            _FONT_CACHE[size] = ImageFont.load_default()
    return _FONT_CACHE[size]


def _load_bold_font(size: int) -> ImageFont.FreeTypeFont:
    key = f"bold_{size}"
    if key not in _FONT_CACHE:
        for path, idx in [
            ("/System/Library/Fonts/HelveticaNeue.ttc", 1),
            ("/System/Library/Fonts/ArialHB.ttc", 0),
            ("/System/Library/Fonts/Helvetica.ttc", 1),
            ("/System/Library/Fonts/Arial.ttf", 0),
            ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 0),
        ]:
            if os.path.exists(path):
                try:
                    _FONT_CACHE[key] = ImageFont.truetype(path, size, index=idx)
                    break
                except Exception:
                    continue
        if key not in _FONT_CACHE:
            _FONT_CACHE[key] = _load_font(size)
    return _FONT_CACHE[key]


def _wrap_text(words: list[str], font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    lines, current = [], ""
    for word in words:
        test = (current + " " + word).strip()
        try:
            w = font.getbbox(test)[2]
        except Exception:
            w = len(test) * 20
        if w > max_width and current:
            lines.append(current)
            current = word
        else:
            current = test
    if current:
        lines.append(current)
    return lines or [""]


def _wrap_text_str(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    return _wrap_text(text.strip().split(), font, max_width)


def _dark_gradient_overlay(H: int, W: int, top_frac: float,
                            rgb: tuple = (8, 8, 8), max_alpha: int = 235) -> Image.Image:
    ov_top = int(H * top_frac)
    arr    = np.zeros((H, W, 4), dtype=np.uint8)
    n      = H - ov_top
    if n > 0:
        alphas = np.clip(np.linspace(0, max_alpha, n), 0, 255).astype(np.uint8)
        arr[ov_top:, :, 0] = rgb[0]
        arr[ov_top:, :, 1] = rgb[1]
        arr[ov_top:, :, 2] = rgb[2]
        arr[ov_top:, :, 3] = alphas[:, None]
    return Image.fromarray(arr, mode="RGBA")


def _full_dark_overlay(H: int, W: int,
                        rgb: tuple = (0, 0, 0),
                        alpha_top: int = 140, alpha_bot: int = 220) -> Image.Image:
    arr    = np.zeros((H, W, 4), dtype=np.uint8)
    alphas = np.clip(np.linspace(alpha_top, alpha_bot, H), 0, 255).astype(np.uint8)
    arr[:, :, 0] = rgb[0]
    arr[:, :, 1] = rgb[1]
    arr[:, :, 2] = rgb[2]
    arr[:, :, 3] = alphas[:, None]
    return Image.fromarray(arr, mode="RGBA")


def _kws_for_bullet(text: str) -> list:
    try:
        from tools.scene_tools import detect_keywords
        return detect_keywords(text)
    except ImportError:
        return []


def _line_kws(kws: list, char_offset: int, line: str) -> list:
    hi = char_offset + len(line)
    return [(s - char_offset, e - char_offset, kt, kc)
            for s, e, kt, kc in kws if s >= char_offset and e <= hi]


def _draw_rich(draw, text: str, x: int, y: int, font, default_color: tuple, kws: list):
    try:
        from tools.scene_tools import draw_rich_text
        draw_rich_text(draw, text, x, y, font, default_color, kws)
    except ImportError:
        draw.text((x, y), text, fill=(*default_color[:3], 255), font=font)


def _render_heading_frame(img_arr: np.ndarray, heading: str, source_name: str, rank: int) -> np.ndarray:
    H, W  = VID_H, VID_W
    style = rank % 4
    WARM  = (255, 250, 235, 255)
    GOLD  = (255, 200, 50,  255)
    TEAL  = (70,  210, 185, 255)

    if style == 0:
        img  = Image.fromarray(img_arr).convert("RGBA").resize((W, H), Image.LANCZOS)
        img  = Image.alpha_composite(img, _dark_gradient_overlay(H, W, 0.30))
        draw = ImageDraw.Draw(img)
        bar_y = int(H * 0.30) + 10
        draw.rectangle([(48, bar_y), (220, bar_y + 4)], fill=GOLD)
        font_h = _load_bold_font(86)
        lines  = _wrap_text(heading.upper().split(), font_h, W - 100)
        y = bar_y + 24
        for line in lines[:3]:
            draw.text((48, y), line, fill=WARM, font=font_h)
            y += 94
        return np.array(img.convert("RGB"))

    elif style == 1:
        img  = Image.fromarray(img_arr).convert("RGBA").resize((W, H), Image.LANCZOS)
        flat = Image.new("RGBA", (W, H), (8, 10, 28, 150))
        img  = Image.alpha_composite(img, flat)
        draw = ImageDraw.Draw(img)
        draw.rectangle([(0, 0), (W, 6)], fill=GOLD)
        font_h  = _load_bold_font(80)
        lines   = _wrap_text(heading.upper().split(), font_h, W - 100)
        total_h = min(len(lines), 4) * 90
        start_y = max(60, (H - total_h) // 2 - 60)
        for line in lines[:4]:
            draw.text((48, start_y), line, fill=WARM, font=font_h)
            start_y += 90
        return np.array(img.convert("RGB"))

    elif style == 2:
        img  = Image.fromarray(img_arr).convert("RGBA").resize((W, H), Image.LANCZOS)
        flat = Image.new("RGBA", (W, H), (0, 0, 0, 165))
        img  = Image.alpha_composite(img, flat)
        draw = ImageDraw.Draw(img)
        draw.rectangle([(0, 0), (W, 5)], fill=TEAL)
        font_h = _load_bold_font(94)
        lines  = _wrap_text(heading.split(), font_h, W - 96)
        y = 48
        for line in lines[:4]:
            draw.text((48, y), line, fill=WARM, font=font_h)
            y += 102
        return np.array(img.convert("RGB"))

    else:
        img  = Image.fromarray(img_arr).convert("RGBA").resize((W, H), Image.LANCZOS)
        img  = Image.alpha_composite(img, _full_dark_overlay(H, W, alpha_top=90, alpha_bot=180))
        draw = ImageDraw.Draw(img)
        draw.rectangle([(0, 0), (W, 5)], fill=GOLD)
        font_h = _load_bold_font(84)
        lines  = _wrap_text(heading.upper().split(), font_h, W - 80)
        y = 36
        for line in lines[:4]:
            draw.text((40, y), line, fill=WARM, font=font_h)
            y += 92
        return np.array(img.convert("RGB"))


def _render_bullets_frame(
    img_arr:   np.ndarray,
    bullets:   list[str],
    slide_idx: int,
    rank:      int,
    n_visible: int,
    anim_t:    float = 1.0,
) -> np.ndarray:
    H, W   = VID_H, VID_W
    style  = rank % 4
    LINE_H = 64
    font_b = _load_bold_font(54)
    WARM   = (255, 250, 235)
    GOLD   = (255, 200, 50)
    TEAL   = (70,  210, 185)

    n_vis = max(0, min(n_visible, len(bullets)))

    def bullet_color(i: int, base_rgb: tuple = (255, 250, 235)) -> tuple:
        if i < n_vis - 1:
            return base_rgb
        t = min(1.0, max(0.0, anim_t))
        if t < 0.18:
            flash = 1.0 - t / 0.18
            return tuple(min(255, int(c + (255 - c) * flash * 0.85)) for c in base_rgb)
        return base_rgb

    if style == 0:
        img  = Image.fromarray(img_arr).convert("RGBA").resize((W, H), Image.LANCZOS)
        flat = Image.new("RGBA", (W, H), (0, 0, 0, 165))
        img  = Image.alpha_composite(img, flat)
        draw = ImageDraw.Draw(img)
        draw.rectangle([(0, 0), (W, 5)], fill=(255, 200, 50, 255))
        y = 58
        for i, bullet in enumerate(bullets[:n_vis]):
            col   = bullet_color(i)
            gcol  = bullet_color(i, GOLD)
            kws   = _kws_for_bullet(bullet)
            lines = _wrap_text_str(bullet, font_b, W - 116)
            draw.ellipse([(48, y + 16), (68, y + 36)], fill=gcol)
            char_off = 0
            for j, line in enumerate(lines[:3]):
                lk = _line_kws(kws, char_off, line)
                if lk:
                    _draw_rich(draw, line, 84, y + j * LINE_H, font_b, col, lk)
                else:
                    draw.text((84, y + j * LINE_H), line, fill=(*col[:3], 255), font=font_b)
                char_off += len(line) + 1
            y += len(lines[:3]) * LINE_H + 26

    elif style == 1:
        img  = Image.fromarray(img_arr).convert("RGBA").resize((W, H), Image.LANCZOS)
        flat = Image.new("RGBA", (W, H), (5, 8, 25, 175))
        img  = Image.alpha_composite(img, flat)
        draw = ImageDraw.Draw(img)
        draw.rectangle([(0, 0), (W, 6)], fill=(255, 200, 50, 255))
        y = 58
        for i, bullet in enumerate(bullets[:n_vis]):
            kws   = _kws_for_bullet(bullet)
            lines = _wrap_text_str(bullet, font_b, W - 96)
            char_off = 0
            for j, line in enumerate(lines[:3]):
                base = GOLD if j == 0 else WARM
                col  = bullet_color(i, base)
                lk   = _line_kws(kws, char_off, line)
                if lk:
                    _draw_rich(draw, line, 48, y + j * LINE_H, font_b, col, lk)
                else:
                    draw.text((48, y + j * LINE_H), line, fill=(*col[:3], 255), font=font_b)
                char_off += len(line) + 1
            y += len(lines[:3]) * LINE_H + 28

    elif style == 2:
        img  = Image.fromarray(img_arr).convert("RGBA").resize((W, H), Image.LANCZOS)
        flat = Image.new("RGBA", (W, H), (0, 0, 0, 165))
        img  = Image.alpha_composite(img, flat)
        draw = ImageDraw.Draw(img)
        draw.rectangle([(0, 0), (W, 5)], fill=(70, 210, 185, 255))
        y = 58
        for i, bullet in enumerate(bullets[:n_vis]):
            col  = bullet_color(i)
            tcol = bullet_color(i, TEAL)
            kws  = _kws_for_bullet(bullet)
            lines = _wrap_text_str(bullet, font_b, W - 104)
            draw.rectangle([(48, y + 4), (52, y + 52)], fill=tcol)
            char_off = 0
            for j, line in enumerate(lines[:3]):
                lk = _line_kws(kws, char_off, line)
                if lk:
                    _draw_rich(draw, line, 64, y + j * LINE_H, font_b, col, lk)
                else:
                    draw.text((64, y + j * LINE_H), line, fill=(*col[:3], 255), font=font_b)
                char_off += len(line) + 1
            y += len(lines[:3]) * LINE_H + 26

    else:
        img  = Image.fromarray(img_arr).convert("RGBA").resize((W, H), Image.LANCZOS)
        img  = Image.alpha_composite(img, _full_dark_overlay(H, W, alpha_top=130, alpha_bot=180))
        draw = ImageDraw.Draw(img)
        draw.rectangle([(0, 0), (W, 5)], fill=(255, 200, 50, 255))
        y = 55
        for i, bullet in enumerate(bullets[:n_vis]):
            col  = bullet_color(i)
            kws  = _kws_for_bullet(bullet)
            lines = _wrap_text_str(bullet, font_b, W - 90)
            char_off = 0
            for j, line in enumerate(lines[:3]):
                lk = _line_kws(kws, char_off, line)
                if lk:
                    _draw_rich(draw, line, 40, y + j * LINE_H, font_b, col, lk)
                else:
                    draw.text((40, y + j * LINE_H), line, fill=(*col[:3], 255), font=font_b)
                char_off += len(line) + 1
            y += len(lines[:3]) * LINE_H + 28

    return np.array(img.convert("RGB"))


HOOK_DUR = 1.25


def create_news_video(
    image_paths:  list[str],
    heading:      str,
    bullets:      list[str],
    source_name:  str,
    music_path:   str | None,
    output_path:  str,
    rank:         int        = 0,
    fps:          int        = 24,
    voice_path:   str | None = None,
    voice_timing: dict | None = None,
    hook:         str | None = None,
) -> str:
    from moviepy.editor import ImageSequenceClip, AudioFileClip, concatenate_audioclips

    FLASH_FRAMES = max(2, int(fps * 0.083))
    W, H         = VID_W, VID_H
    n            = len(bullets)
    split        = max(1, n // 2)
    bullets_a    = bullets[:split]
    bullets_b    = bullets[split:]

    if voice_timing and voice_timing.get("bullet_starts"):
        bs    = voice_timing["bullet_starts"]
        h_end = voice_timing.get("heading_end", 4.0)
        tot   = voice_timing.get("total_duration", 30.0)
        s1    = max(4.5, h_end + 0.5)
        s2    = max(5.0, bs[split] - s1 + 0.8) if split < len(bs) else max(5.0, (tot - s1) * 0.5)
        s3    = max(5.0, tot - s1 - s2 + 1.5)
        SLIDE_SECS = [s1, s2, s3]
    else:
        w_h   = max(1, len(heading.split()))
        w_a   = max(1, sum(len(b.split()) for b in bullets_a))
        w_b   = max(1, sum(len(b.split()) for b in bullets_b))
        tot_w = w_h + w_a + w_b
        base  = 30.0
        s1 = max(4.5, base * w_h / tot_w)
        s2 = max(5.0, base * w_a / tot_w)
        s3 = max(5.0, base - s1 - s2)
        SLIDE_SECS = [s1, s2, s3]

    if hook:
        SLIDE_SECS[0] += HOOK_DUR

    total_dur = sum(SLIDE_SECS)

    _FALLBACK_GRADIENTS = [
        ((25, 45, 90),  (5,  10, 20)),
        ((15, 15, 55),  (5,  5,  18)),
        ((14, 14, 14),  (5,  5,   5)),
        ((40, 12, 12),  (8,  8,   8)),
    ]

    def load_img(path, slide_idx=0):
        if path and os.path.exists(path):
            return np.array(Image.open(path).convert("RGB").resize((W, H), Image.LANCZOS))
        top, bot = _FALLBACK_GRADIENTS[rank % 4]
        t   = np.linspace(0, 1, H)[:, np.newaxis]
        arr = np.zeros((H, W, 3), dtype=np.float32)
        for c in range(3):
            arr[:, :, c] = top[c] * (1 - t) + bot[c] * t
        return arr.astype(np.uint8)

    imgs = [load_img(p, i) for i, p in enumerate((image_paths + [None, None, None])[:3])]

    try:
        from tools.scene_tools import (
            classify_scene, SCENE_CAMERA,
            apply_camera_move, estimate_word_timing, render_word_captions,
        )
        _scene_ok = True
    except ImportError:
        _scene_ok = False

    slide_scenes = ["push_forward", "pan_left", "zoom_in"]
    if _scene_ok:
        if bullets_a:
            slide_scenes[1] = SCENE_CAMERA.get(classify_scene(bullets_a[0], 1, 4), "pan_left")
        if bullets_b:
            slide_scenes[2] = SCENE_CAMERA.get(classify_scene(bullets_b[-1], 3, 4), "zoom_in")

    all_word_timing: list = []
    if _scene_ok and voice_timing:
        bs       = voice_timing.get("bullet_starts", [])
        h_end    = voice_timing.get("heading_end", 4.0)
        tot      = voice_timing.get("total_duration", 30.0)
        h_offset = HOOK_DUR if hook else 0.0

        all_word_timing.extend(estimate_word_timing(heading, h_offset, h_offset + h_end))
        for i, b in enumerate(bullets):
            b_start = (bs[i] if i < len(bs) else h_end + i * 2.0) + h_offset
            b_end   = (bs[i + 1] if i + 1 < len(bs) else tot) + h_offset
            all_word_timing.extend(estimate_word_timing(b, b_start, b_end))

    print(f"  Rendering news video: {heading[:50]}  "
          f"[{SLIDE_SECS[0]:.1f}s / {SLIDE_SECS[1]:.1f}s / {SLIDE_SECS[2]:.1f}s]"
          + (f"  hook={hook[:30]!r}" if hook else "")
          + (f"  scene_cam={slide_scenes[1]}/{slide_scenes[2]}" if _scene_ok else ""))
    all_frames = []

    slide_starts = [0.0, SLIDE_SECS[0], SLIDE_SECS[0] + SLIDE_SECS[1]]

    def _n_vis_at(t_abs: float, bullet_list: list, bullet_timestamps: list) -> tuple[int, float]:
        n = 0
        for ts in bullet_timestamps[:len(bullet_list)]:
            if ts <= t_abs:
                n += 1
        if n == 0:
            return 0, 0.0
        last_ts = bullet_timestamps[n - 1]
        anim_t  = min(1.0, (t_abs - last_ts) / 0.20)
        return n, anim_t

    bt_a       = (voice_timing.get("bullet_starts", [])[:split] if voice_timing else None)
    bt_b       = (voice_timing.get("bullet_starts", [])[split:] if voice_timing else None)
    PUNCH_DUR  = 0.35
    hook_frames = int(HOOK_DUR * fps) if hook else 0

    for seg_idx, (img_arr, seg_dur) in enumerate(zip(imgs, SLIDE_SECS)):
        seg_frames = int(seg_dur * fps)

        for fi in range(seg_frames):
            abs_t      = slide_starts[seg_idx] + fi / fps
            progress   = fi / max(seg_frames - 1, 1)
            t_in_slide = fi / fps

            if _scene_ok:
                img_kb = apply_camera_move(img_arr, slide_scenes[seg_idx], t_in_slide, SLIDE_SECS[seg_idx])
            else:
                kb_zoom = 1.0 + 0.08 * (progress if seg_idx % 2 == 0 else 1.0 - progress)
                kb_cx   = 0.5 + 0.02 * progress * (1 if seg_idx % 4 < 2 else -1)
                img_kb  = _zoom_frame_pan(img_arr, kb_zoom, kb_cx)

            if seg_idx == 0:
                if hook and fi < hook_frames:
                    frame = _render_hook_card(img_kb, hook, t_in_slide)
                    frames_left = hook_frames - fi
                    if frames_left <= FLASH_FRAMES:
                        flash_t = 1.0 - frames_left / FLASH_FRAMES
                        frame   = (frame.astype(np.float32) * (1 - flash_t * 0.65)
                                   + 255 * flash_t * 0.65).clip(0, 255).astype(np.uint8)
                else:
                    frame = _render_heading_frame(img_kb, heading, source_name, rank)
                    t_since_hook = t_in_slide - (HOOK_DUR if hook else 0)
                    if t_since_hook < PUNCH_DUR:
                        punch_scale = 1.0 + 0.15 * max(0.0, 1.0 - t_since_hook / PUNCH_DUR)
                        frame = _apply_punch_scale(frame, punch_scale)

            elif seg_idx == 1:
                if bt_a:
                    nv, at = _n_vis_at(abs_t, bullets_a, bt_a)
                    nv     = max(1, nv) if abs_t >= slide_starts[1] else 0
                else:
                    nv = max(1, int(progress * len(bullets_a)) + 1)
                    at = 1.0
                frame = _render_bullets_frame(img_kb, bullets_a, 0, rank, nv, at)

            else:
                if bt_b:
                    nv, at = _n_vis_at(abs_t, bullets_b, bt_b)
                    nv     = max(1, nv) if abs_t >= slide_starts[2] else 0
                else:
                    nv = max(1, int(progress * len(bullets_b)) + 1)
                    at = 1.0
                frame = _render_bullets_frame(img_kb, bullets_b, 1, rank, nv, at)

            if all_word_timing:
                is_hook_card = (seg_idx == 0 and hook and fi < hook_frames)
                if not is_hook_card:
                    frame = render_word_captions(
                        frame, all_word_timing, abs_t,
                        caption_y=H - 140, font_size=38, style=rank % 3,
                    )

            if seg_idx < 2 and fi >= seg_frames - FLASH_FRAMES:
                flash_t = (fi - (seg_frames - FLASH_FRAMES)) / FLASH_FRAMES
                frame   = (frame.astype(np.float32) * (1 - flash_t * 0.65)
                           + 255 * flash_t * 0.65).clip(0, 255).astype(np.uint8)

            all_frames.append(frame)

    clip = ImageSequenceClip(all_frames, fps=fps)

    from moviepy.editor import CompositeAudioClip
    audio_tracks = []

    if voice_path and os.path.exists(voice_path):
        print("  Adding voice narration...")
        voice_clip = AudioFileClip(voice_path)
        if voice_clip.duration > total_dur:
            voice_clip = voice_clip.subclip(0, total_dur)
        if hook:
            voice_clip = voice_clip.set_start(HOOK_DUR)
        voice_clip = voice_clip.audio_fadein(0.3).audio_fadeout(1.0)
        audio_tracks.append(voice_clip)

    if music_path and os.path.exists(music_path):
        print("  Adding background music...")
        music_clip = AudioFileClip(music_path)
        if music_clip.duration < total_dur:
            music_clip = concatenate_audioclips(
                [music_clip] * math.ceil(total_dur / music_clip.duration)
            )
        music_clip = music_clip.subclip(0, total_dur).audio_fadein(1.5).audio_fadeout(2)
        vol = 0.30 if audio_tracks else 1.0
        music_clip = music_clip.volumex(vol)
        audio_tracks.append(music_clip)

    if audio_tracks:
        mixed = CompositeAudioClip(audio_tracks)
        clip  = clip.set_audio(mixed)
    else:
        print("  No audio — silent video")

    print(f"  Saving → {os.path.basename(output_path)}")
    clip.write_videofile(
        output_path,
        fps=fps,
        codec="libx264",
        audio_codec="aac",
        temp_audiofile=str(Path(output_path).parent / f"_tmp_{Path(output_path).stem}.m4a"),
        remove_temp=True,
        logger=None,
    )
    return output_path
