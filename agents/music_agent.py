import os
import time
import requests
from dotenv import load_dotenv
from agents.base_agent import BaseAgent, AgentResult
from tools.video_tools import NICHE_MUSIC_QUERIES, download_music, MUSIC_DIR

load_dotenv(override=True)

_STOP_WORDS = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "by","is","are","was","were","has","have","had","this","that","these",
    "those","it","its","be","as","do","did","will","can","may","how","why",
    "what","when","where","who",
}

MOOD_QUERIES = {
    "curiosity":    "curious ambient electronic instrumental",
    "inspiration":  "inspirational uplifting background music",
    "excitement":   "energetic upbeat electronic music",
    "humor":        "playful quirky upbeat music",
    "wonder":       "cinematic wonder ambient orchestral",
}


def _topic_keywords(topic: str) -> list[str]:
    words = [w.strip(".,!?\"'()[]") for w in topic.split()]
    return [w for w in words if w.lower() not in _STOP_WORDS and len(w) > 3][:4]


def _freesound_search(query: str, rank: int, api_key: str) -> dict | None:
    try:
        r = requests.get(
            "https://freesound.org/apiv2/search/text/",
            params={
                "query":     query,
                "token":     api_key,
                "fields":    "id,name,previews,duration,username",
                "filter":    "duration:[20 TO 120]",
                "page_size": 15,
                "sort":      "rating_desc",
            },
            timeout=10,
        )
        results = [t for t in r.json().get("results", [])
                   if t.get("previews", {}).get("preview-hq-mp3")]
        return results[rank % len(results)] if results else None
    except Exception:
        return None


def _find_track(topic: str, niche: str, emotion: str, rank: int) -> dict | None:
    api_key = os.getenv("FREESOUND_API_KEY", "")
    if not api_key:
        return None

    niche_q = NICHE_MUSIC_QUERIES.get(niche.lower(), "ambient background music")

    # 1. Topic keywords + niche style (most relevant)
    keywords = _topic_keywords(topic)
    if keywords:
        q = " ".join(keywords[:3]) + " " + niche_q
        track = _freesound_search(q, rank, api_key)
        if track:
            return track

    # 2. Mood / emotion-based query
    mood_q = MOOD_QUERIES.get(emotion.lower(), "")
    if mood_q:
        track = _freesound_search(mood_q, rank, api_key)
        if track:
            return track

    # 3. Pure niche fallback
    return _freesound_search(niche_q, rank, api_key)


class MusicAgent(BaseAgent):

    def __init__(self, niche: str = "technology"):
        super().__init__(name="MusicAgent", model="fast")
        self.niche     = niche
        self.api_key   = os.getenv("FREESOUND_API_KEY", "")

    @property
    def system_prompt(self) -> str:
        return "You select background music for social media videos."

    def run(self, input_data: dict) -> AgentResult:
        posts = input_data.get("posts", [])
        niche = input_data.get("niche", self.niche)

        if not posts:
            return self._failure("No posts received.")

        if not self.api_key:
            self._log("[yellow]FREESOUND_API_KEY not set — music skipped[/yellow]")
            for post in posts:
                post.setdefault("music_path",  None)
                post.setdefault("music_title", "No music")
            return self._success(
                data={"posts": posts, "music_added": 0, "total": len(posts)},
                reasoning="No Freesound API key",
            )

        self._log(
            f"Fetching tracks for {len(posts)} post(s)  |  "
            f"niche: [cyan]{niche}[/cyan]  |  "
            f"strategy: topic-keywords → mood → niche"
        )

        _fallback_path  = None
        _fallback_title = "No music"
        music_added     = 0

        for post in posts:
            rank    = post.get("rank", 0)
            topic   = post.get("topic", "")
            emotion = post.get("emotion", "")
            ts      = int(time.time() * 1000)

            self._log(f"  Searching: [cyan]{topic[:50]}[/cyan]  emotion={emotion}")
            track = _find_track(topic, niche, emotion, rank)

            if track:
                title    = f"{track.get('name', '?')} — {track.get('username', '?')}"
                filename = f"music_{niche}_{rank}_{ts}.mp3"
                path     = download_music(track, filename)
                if path:
                    post["music_path"]  = path
                    post["music_title"] = title
                    _fallback_path      = path
                    _fallback_title     = title
                    music_added        += 1
                    self._log(f"  [green]✓ {title[:60]}[/green]")
                    continue

            if _fallback_path:
                post["music_path"]  = _fallback_path
                post["music_title"] = _fallback_title
                self._log(f"  Reusing cached track for post#{rank}")
            else:
                post["music_path"]  = None
                post["music_title"] = "No music"
                self._log(f"  [yellow]No track found for post#{rank}[/yellow]")

        self._log(f"[green]{music_added}/{len(posts)} unique track(s) fetched[/green]")
        return self._success(
            data={"posts": posts, "music_added": music_added, "total": len(posts)},
            reasoning=f"{music_added}/{len(posts)} Freesound tracks | topic+mood+niche strategy",
        )
