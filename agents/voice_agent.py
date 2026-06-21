import os
import time
from pathlib import Path
from dotenv import load_dotenv
from agents.base_agent import BaseAgent, AgentResult
from tools.voice_tools import (
    get_voice_for_rank,
    build_synced_narration,
    build_standard_narration,
    VOICE_DIR,
    EDGE_VOICES,
)

load_dotenv(override=True)


class VoiceAgent(BaseAgent):

    def __init__(self, voice_preset: str | None = None):
        super().__init__(name="VoiceAgent", model="fast")
        self.base_preset = voice_preset or os.getenv("VOICE_PRESET", "en-US")
        parts = self.base_preset.split("-")
        self.accent = "-".join(parts[:2]) if len(parts) >= 2 else "en-US"

    @property
    def system_prompt(self) -> str:
        return "You are a voice narration assistant."

    def run(self, input_data: dict) -> AgentResult:
        posts = input_data.get("posts", [])
        niche = input_data.get("niche", "latest_news")

        if not posts:
            return self._failure("No posts to narrate.")

        self._log(f"Generating synced voice narration for [bold]{len(posts)}[/bold] posts")

        ok = 0
        for post in posts:
            rank  = post.get("rank", 1)
            topic = post.get("topic", "")
            voice, preset_key = get_voice_for_rank(rank, self.accent + "-male")
            gender = "female" if rank % 2 == 0 else "male"

            self._log(
                f"  Post #{rank} — [cyan]{topic[:50]}[/cyan] "
                f"[dim]({gender} voice: {voice})[/dim]"
            )

            ts = int(time.time())
            out_path = str(VOICE_DIR / f"voice_{rank}_{ts}.mp3")

            try:
                is_news = bool(post.get("bullets"))

                if is_news:
                    heading = post.get("heading", topic)
                    bullets = post.get("bullets", [])[:10]
                    self._log(f"    Narrating heading + {len(bullets)} bullets (per-segment sync)…")

                    voice_path, timing = build_synced_narration(
                        heading    = heading,
                        bullets    = bullets,
                        output_path = out_path,
                        voice      = voice,
                    )

                    if voice_path and timing:
                        post["voice_path"]   = voice_path
                        post["voice_timing"] = timing
                        post["voice_preset"] = preset_key
                        dur = timing["total_duration"]
                        n_ts = len(timing["bullet_starts"])
                        self._log(
                            f"    [green]Synced voice ready — {dur:.1f}s total, "
                            f"{n_ts} bullet timestamps[/green]"
                        )
                        ok += 1
                    else:
                        self._log("    [yellow]Voice failed — video will be silent[/yellow]")
                        post["voice_path"]   = None
                        post["voice_timing"] = {}

                else:
                    self._log("    Narrating full caption (standard post)…")
                    voice_path = build_standard_narration(post, niche, voice)
                    post["voice_path"]   = voice_path
                    post["voice_timing"] = {}
                    post["voice_preset"] = preset_key
                    if voice_path:
                        self._log(f"    [green]Voice ready → {Path(voice_path).name}[/green]")
                        ok += 1
                    else:
                        self._log("    [yellow]Voice failed — video will be silent[/yellow]")

            except Exception as e:
                self._log(f"    [red]Voice error: {e}[/red]")
                post["voice_path"]   = None
                post["voice_timing"] = {}

        self._log(f"[green]Voice: {ok}/{len(posts)} narrations ready[/green]")

        return self._success(
            data={"posts": posts, "total": len(posts), "voice_ok": ok},
            reasoning=f"Narrated {ok}/{len(posts)} posts with synced timestamps",
        )
