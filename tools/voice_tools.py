import asyncio
import os
import re
import subprocess
import tempfile
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

VOICE_DIR = Path(__file__).parent.parent / "generated_voices"
VOICE_DIR.mkdir(exist_ok=True)

ELEVEN_VOICES = {
    "male": [
        ("onwK4e9ZLuTAKqWW03F9", "Daniel"),
        ("nPczCjzI2devNBz1zQrb", "Brian"),
        ("JBFqnCBsd6RMkjVDRZzb", "George"),
    ],
    "female": [
        ("EXAVITQu4vr4xnSDxMaL", "Sarah"),
        ("XrExE9yKIg1WjnnlVkGX", "Matilda"),
        ("Xb7hH8MSUJpSbSDYk0k2", "Alice"),
    ],
}

EDGE_VOICES = {
    "en-US-male":   "en-US-GuyNeural",
    "en-US-female": "en-US-JennyNeural",
    "en-GB-male":   "en-GB-RyanNeural",
    "en-GB-female": "en-GB-SoniaNeural",
    "en-IN-male":   "en-IN-PrabhatNeural",
    "en-IN-female": "en-IN-NeerjaNeural",
    "en-AU-male":   "en-AU-WilliamNeural",
    "en-AU-female": "en-AU-NatashaNeural",
}

DEFAULT_PRESET = os.getenv("VOICE_PRESET", "en-US-male")


def get_voice_for_rank(rank: int, base_preset: str = None) -> tuple[str, str]:
    gender = "male" if rank % 2 == 1 else "female"

    eleven_key = os.getenv("ELEVENLABS_API_KEY", "")
    if eleven_key and eleven_key != "paste_your_key_here":
        voice_list = ELEVEN_VOICES[gender]
        idx        = (rank // 2) % len(voice_list)
        voice_id, voice_name = voice_list[idx]
        return voice_id, f"elevenlabs-{gender}-{voice_name.lower()}"

    preset = base_preset or DEFAULT_PRESET
    parts  = preset.split("-")
    accent = "-".join(parts[:2]) if len(parts) >= 2 else "en-US"
    key    = f"{accent}-{gender}"
    if key not in EDGE_VOICES:
        key = "en-US-male" if gender == "male" else "en-US-female"
    return EDGE_VOICES[key], key



def synthesize_elevenlabs(
    text:        str,
    output_path: str,
    voice_id:    str,
    model:       str = "eleven_turbo_v2",
) -> str | None:
    api_key = os.getenv("ELEVENLABS_API_KEY", "")
    if not api_key or api_key == "paste_your_key_here":
        return None

    import requests
    url     = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "Accept":       "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key":   api_key,
    }
    payload = {
        "text":     text,
        "model_id": model,
        "voice_settings": {
            "stability":         0.45,
            "similarity_boost":  0.75,
            "style":             0.25,
            "use_speaker_boost": True,
        },
    }
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=30)
        if r.status_code == 200:
            with open(output_path, "wb") as f:
                f.write(r.content)
            _normalize_audio(output_path)
            print(f"  [Voice] ElevenLabs ({voice_id[:8]}…) → {Path(output_path).name}")
            return output_path
        else:
            print(f"  [Voice] ElevenLabs {r.status_code}: {r.text[:150]}")
    except Exception as e:
        print(f"  [Voice] ElevenLabs failed: {e}")
    return None



async def _edge_async(text: str, path: str, voice: str, rate: str) -> bool:
    try:
        import edge_tts
        await edge_tts.Communicate(text, voice, rate=rate).save(path)
        return os.path.exists(path) and os.path.getsize(path) > 500
    except Exception as e:
        print(f"  [Voice] edge-tts error: {e}")
        return False


def synthesize_edge(
    text: str,
    output_path: str,
    voice: str = "en-US-GuyNeural",
    rate: str = "+8%",
) -> str | None:
    try:
        loop = asyncio.new_event_loop()
        ok   = loop.run_until_complete(_edge_async(text, output_path, voice, rate))
        loop.close()
        if ok:
            _normalize_audio(output_path)
            print(f"  [Voice] EdgeTTS {voice} → {Path(output_path).name}")
            return output_path
    except Exception as e:
        print(f"  [Voice] edge-tts failed: {e}")
    return None



def _synthesize_segment(
    text:        str,
    output_path: str,
    voice:       str,
    rate:        str = "+8%",
) -> str | None:
    eleven_key = os.getenv("ELEVENLABS_API_KEY", "")
    if not voice.startswith("en-") and eleven_key and eleven_key != "paste_your_key_here":
        result = synthesize_elevenlabs(text, output_path, voice)
        if result:
            return result
        edge_voice = "en-US-GuyNeural"
    else:
        edge_voice = voice if voice.startswith("en-") else "en-US-GuyNeural"

    return synthesize_edge(text, output_path, edge_voice, rate)



def _normalize_audio(path: str) -> None:
    tmp = path + "._norm.mp3"
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", path, "-ar", "44100", "-ac", "2", "-q:a", "2", tmp],
        capture_output=True,
    )
    if r.returncode == 0 and os.path.exists(tmp):
        os.replace(tmp, path)


def get_audio_duration(path: str) -> float:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", path],
            capture_output=True, text=True,
        )
        return float(r.stdout.strip())
    except Exception:
        return 3.0


def _silence(duration_s: float, output_path: str) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi",
         "-i", "anullsrc=r=44100:cl=stereo",
         "-t", str(duration_s), "-q:a", "9", output_path],
        capture_output=True,
    )


def _concat_audio(paths: list[str], output_path: str) -> bool:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for p in paths:
            f.write(f"file '{p}'\n")
        lst = f.name
    r = subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
         "-i", lst, "-ar", "44100", "-ac", "2", "-q:a", "2", output_path],
        capture_output=True,
    )
    try:
        os.remove(lst)
    except Exception:
        pass
    return r.returncode == 0 and os.path.exists(output_path)



def build_synced_narration(
    heading:       str,
    bullets:       list[str],
    output_path:   str,
    voice:         str,
    pause_heading: float = 0.50,
    pause_bullet:  float = 0.32,
    rate:          str   = "+8%",
) -> tuple[str | None, dict]:
    tmp = Path(tempfile.mkdtemp())
    segments: list[str] = []
    timing = {
        "heading_start": 0.0,
        "heading_end":   0.0,
        "bullet_starts": [],
        "total_duration": 0.0,
    }

    h_clean = _clean(heading)
    h_path  = str(tmp / "h.mp3")
    if not _synthesize_segment(h_clean, h_path, voice, rate):
        return None, {}
    segments.append(h_path)

    h_dur = get_audio_duration(h_path)
    timing["heading_end"] = h_dur

    p0 = str(tmp / "ph.mp3")
    _silence(pause_heading, p0)
    segments.append(p0)
    current_t = h_dur + pause_heading

    for i, bullet in enumerate(bullets):
        b_clean = _clean(bullet)
        if not b_clean:
            timing["bullet_starts"].append(current_t)
            continue

        b_path = str(tmp / f"b{i}.mp3")
        if _synthesize_segment(b_clean, b_path, voice, rate):
            timing["bullet_starts"].append(current_t)
            segments.append(b_path)
            b_dur = get_audio_duration(b_path)
            current_t += b_dur
        else:
            timing["bullet_starts"].append(current_t)

        if i < len(bullets) - 1:
            pb = str(tmp / f"pb{i}.mp3")
            _silence(pause_bullet, pb)
            segments.append(pb)
            current_t += pause_bullet

    outro = _clean("Follow for more stories like this every day.")
    op    = str(tmp / "outro.mp3")
    pp    = str(tmp / "pause_outro.mp3")
    _silence(pause_heading, pp)
    segments.append(pp)
    current_t += pause_heading
    if _synthesize_segment(outro, op, voice, rate):
        segments.append(op)
        current_t += get_audio_duration(op)

    if not _concat_audio(segments, output_path):
        return None, {}

    for s in segments:
        try:
            os.remove(s)
        except Exception:
            pass
    try:
        tmp.rmdir()
    except Exception:
        pass

    timing["total_duration"] = get_audio_duration(output_path)
    return output_path, timing



def build_standard_narration(post: dict, niche: str, voice: str) -> str | None:
    caption = ""
    if "instagram" in post:
        caption = post["instagram"].get("caption", "")
    if not caption:
        caption = post.get("description", post.get("topic", ""))
    lines  = [l for l in caption.split("\n")
              if l.strip() and not l.strip().startswith("#")
              and not all(w.startswith("#") for w in l.split())]
    script = _clean(" ".join(lines) + " Follow for more.")
    words  = script.split()
    if len(words) > 80:
        script = " ".join(words[:80]).rstrip(",") + "."
    out = str(VOICE_DIR / f"voice_std_{id(post)}_{os.getpid()}.mp3")
    return _synthesize_segment(script, out, voice)



def _gtts_fallback(text: str, output_path: str) -> str | None:
    try:
        from gtts import gTTS
        tts = gTTS(text=text, lang="en", slow=False)
        tmp = output_path + ".raw.mp3"
        tts.save(tmp)
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", tmp,
             "-ar", "44100", "-ac", "2", "-q:a", "2", output_path],
            capture_output=True,
        )
        try:
            os.remove(tmp)
        except Exception:
            pass
        if r.returncode == 0:
            print(f"  [Voice] gTTS fallback → {Path(output_path).name}")
            return output_path
    except Exception as e:
        print(f"  [Voice] gTTS failed: {e}")
    return None



def _clean(text: str) -> str:
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"\*+", "", text)
    text = re.sub(
        r"[\U00010000-\U0010FFFF\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
        r"\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]+",
        "", text, flags=re.UNICODE,
    )
    return re.sub(r"\s+", " ", text).strip()
