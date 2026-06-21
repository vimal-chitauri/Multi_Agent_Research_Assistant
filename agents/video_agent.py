import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from agents.base_agent import BaseAgent, AgentResult
from tools.video_tools import (
    NICHE_VIDEO_PROMPTS,
    search_freesound_music,
    download_music,
    create_video,
    create_news_video,
    group_bullet_points,
)

load_dotenv(override=True)

VIDEOS_DIR = Path(__file__).parent.parent / "generated_videos"
VIDEOS_DIR.mkdir(exist_ok=True)

NICHE_CONTENT_PROMPTS = {
    "latest_news": (
        "You are a news anchor. For the breaking news topic '{topic}', summarize the 8 most important facts "
        "a viewer needs to know right now. Each fact must be under 12 words. Be factual, concise, and neutral. "
        "Respond ONLY with a JSON array of exactly 8 strings: [\"fact1\", ..., \"fact8\"]"
    ),
    "fitness": (
        "You are a certified fitness coach. For the topic '{topic}', generate exactly 8 practical tips. "
        "Mix nutrition, workout, recovery, and mindset tips. "
        "Each tip must be under 10 words. Be specific and actionable. "
        "Respond ONLY with a JSON array of exactly 8 strings: [\"tip1\", ..., \"tip8\"]"
    ),
    "motivation": (
        "You are a motivational coach. For the topic '{topic}', provide: "
        "2 powerful quotes from famous historical figures (include their names), "
        "then 6 short action steps under 10 words each. "
        "Respond ONLY with a JSON array of exactly 8 strings."
    ),
    "technology": (
        "You are a tech expert. For the topic '{topic}', share 8 fascinating facts or tips "
        "that most people don't know. Each must be under 12 words. Focus on practical or surprising info. "
        "Respond ONLY with a JSON array of exactly 8 strings: [\"fact1\", ..., \"fact8\"]"
    ),
    "finance": (
        "You are a financial advisor. For the topic '{topic}', give 8 practical money tips. "
        "Each tip under 10 words. Focus on savings, investing, or avoiding mistakes. "
        "Respond ONLY with a JSON array of exactly 8 strings: [\"tip1\", ..., \"tip8\"]"
    ),
    "crypto": (
        "You are a crypto expert. For the topic '{topic}', share 8 important facts for beginners. "
        "Each under 12 words. Focus on safety, basics, and real information. "
        "Respond ONLY with a JSON array of exactly 8 strings: [\"fact1\", ..., \"fact8\"]"
    ),
    "marketing": (
        "You are a marketing expert. For the topic '{topic}', share 8 powerful tactics that work. "
        "Each under 10 words. Be specific and practical. "
        "Respond ONLY with a JSON array of exactly 8 strings: [\"tip1\", ..., \"tip8\"]"
    ),
    "food": (
        "You are a nutritionist and chef. For the topic '{topic}', share 8 interesting facts or tips. "
        "Mix nutrition and cooking. Each under 10 words. "
        "Respond ONLY with a JSON array of exactly 8 strings: [\"tip1\", ..., \"tip8\"]"
    ),
    "travel": (
        "You are a travel expert. For the topic '{topic}', share 8 practical travel hacks or destination facts. "
        "Each under 12 words. "
        "Respond ONLY with a JSON array of exactly 8 strings: [\"tip1\", ..., \"tip8\"]"
    ),
}

DEFAULT_PROMPT = (
    "For the topic '{topic}' in the {niche} space, share 8 interesting facts or tips. "
    "Each under 12 words. Be informative and engaging. "
    "Respond ONLY with a JSON array of exactly 8 strings: [\"fact1\", ..., \"fact8\"]"
)


class VideoAgent(BaseAgent):

    def __init__(self, duration: int = 30, niche: str = "technology"):
        super().__init__(name="VideoAgent", model="smart")
        self.duration   = duration
        self.niche      = niche
        self.hf_token   = os.getenv("HUGGINGFACE_TOKEN")
        self.freesound  = os.getenv("FREESOUND_API_KEY")

    @property
    def system_prompt(self) -> str:
        return f"You are an expert content creator for {self.niche} social media videos. Always respond in valid JSON."

    def run(self, input_data: dict) -> AgentResult:
        posts = input_data.get("posts", [])
        if not posts:
            return self._failure("No posts received.")

        niche = input_data.get("niche", self.niche)

        if not self.hf_token:
            self._log("[yellow]HUGGINGFACE_TOKEN not set — AI video disabled, using Ken Burns[/yellow]")
        if not self.freesound:
            self._log("[yellow]FREESOUND_API_KEY not set — videos will be silent[/yellow]")

        self._log(
            f"Creating [bold]{self.duration}s[/bold] videos | niche: [cyan]{niche}[/cyan] | "
            f"AI video: [cyan]{'on' if self.hf_token else 'off (Ken Burns fallback)'}[/cyan]"
        )

        videos_created    = 0
        _fallback_music   = None
        _fallback_title   = "No music"

        for post in posts:
            topic      = post["topic"]
            image_path = post.get("image_path")
            rank       = post.get("rank", 0)
            timestamp  = int(time.time())

            self._log(f"Processing: [cyan]{topic}[/cyan]")

            music_path  = None
            music_title = "No music"
            if self.freesound:
                track = search_freesound_music(niche, rank=rank)
                if track:
                    music_title = f"{track.get('name', '?')} — {track.get('username', '?')}"
                    self._log(f"  Music: {music_title}")
                    music_path = download_music(track, f"music_{niche}_{rank}_{timestamp}.mp3")

                if not music_path and _fallback_music:
                    music_path  = _fallback_music
                    music_title = _fallback_title
                    self._log(f"  Music: reusing cached track (Freesound rate-limit)")
                elif not music_path:
                    self._log("  [yellow]No Freesound track found[/yellow]")

                if music_path:
                    _fallback_music = music_path
                    _fallback_title = music_title

            video_path = str(VIDEOS_DIR / f"post_{rank}_{timestamp}.mp4")

            try:
                if niche.lower() == "latest_news":
                    image_paths = post.get("image_paths", [])
                    if image_path and not image_paths:
                        image_paths = [image_path]
                    if not image_paths:
                        self._log("  [yellow]No AI images — using gradient fallback slides[/yellow]")

                    heading = post.get("heading", topic)
                    bullets = post.get("bullets", [])
                    if not bullets:
                        bullets = self._generate_info_content(topic, niche)

                    create_news_video(
                        image_paths  = image_paths,
                        heading      = heading,
                        bullets      = bullets,
                        source_name  = post.get("source_name", ""),
                        music_path   = music_path,
                        output_path  = video_path,
                        rank         = rank,
                        voice_path   = post.get("voice_path"),
                        voice_timing = post.get("voice_timing"),
                        hook         = self._generate_hook(topic, niche),
                    )
                    post.update({
                        "video_path":   video_path,
                        "video_status": "created",
                        "music_title":  music_title,
                    })

                else:
                    if not image_path or not os.path.exists(image_path):
                        self._log("  [yellow]No image — skipping[/yellow]")
                        post.update({"video_path": None, "video_status": "skipped — no image", "music_title": None})
                        continue

                    info_lines   = self._generate_info_content(topic, niche)
                    video_prompt = self._build_video_prompt(topic, niche)
                    groups       = group_bullet_points(info_lines)

                    create_video(
                        image_path   = image_path,
                        title        = topic,
                        info_lines   = info_lines,
                        music_path   = music_path,
                        output_path  = video_path,
                        duration     = self.duration,
                        niche        = niche,
                        voice_path   = post.get("voice_path"),
                        hf_token     = self.hf_token,
                        video_prompt = video_prompt,
                        rank         = rank,
                        hook         = self._generate_hook(topic, niche),
                    )
                    post.update({
                        "video_path":    video_path,
                        "video_status":  "created",
                        "music_title":   music_title,
                        "info_content":  info_lines,
                        "n_segments":    len(groups),
                    })

                videos_created += 1
                self._log(f"  [green]Done: {Path(video_path).name}[/green]")

            except Exception as e:
                self._log(f"  [red]Render failed: {e}[/red]")
                post.update({"video_path": None, "video_status": f"failed — {str(e)[:80]}", "music_title": None})

        self._log(f"[green]{videos_created}/{len(posts)} videos created[/green]")
        return self._success(
            data={
                "posts":         posts,
                "total":         len(posts),
                "videos_created": videos_created,
                "duration_sec":  self.duration,
            },
            reasoning=f"{videos_created}/{len(posts)} x {self.duration}s videos | niche: {niche} | "
                      f"AI video: {'LTX-Video' if self.hf_token else 'Ken Burns fallback'} | "
                      f"music: {'Freesound' if self.freesound else 'silent'}",
        )

    def _generate_info_content(self, topic: str, niche: str) -> list[str]:
        prompt_template = NICHE_CONTENT_PROMPTS.get(niche.lower(), DEFAULT_PROMPT)
        prompt = prompt_template.format(topic=topic, niche=niche)

        try:
            response = self.think(prompt)
            cleaned  = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            lines = json.loads(cleaned.strip())
            if isinstance(lines, list):
                return [str(l) for l in lines[:10]]
        except Exception as e:
            self._log(f"  [yellow]LLM content gen failed ({e}) — using defaults[/yellow]")

        return _fallback_content(niche)

    def _generate_hook(self, topic: str, niche: str) -> str | None:
        prompt = (
            f"You are a viral short-form content writer. For the topic '{topic}' in the "
            f"{niche} niche, write ONE hook under 6 words that creates curiosity or urgency — "
            f"the kind that stops someone mid-scroll. Do NOT state the topic directly. "
            f"No colons, no quotes, no hashtags. Respond with ONLY the hook text."
        )
        try:
            hook = self.think(prompt).strip().strip('"').strip("'")
            words = hook.split()[:12]
            mid   = len(words) // 2
            return " ".join(words[:mid]) + "\n" + " ".join(words[mid:])
        except Exception:
            return None

    def _build_video_prompt(self, topic: str, niche: str) -> str:
        template = NICHE_VIDEO_PROMPTS.get(
            niche.lower(),
            "Professional cinematic scene about {topic}, 4K high quality, smooth motion",
        )
        return template.format(topic=topic)


def _fallback_content(niche: str) -> list[str]:
    fallbacks = {
        "latest_news": ["Breaking news happening around the world now", "Stay informed with the latest updates",
                        "Top stories making headlines today", "Follow for real-time news coverage",
                        "Global events shape our daily lives", "Accurate reporting matters more than speed",
                        "Share news that informs your community", "Knowledge is the first step to change"],
        "fitness":    ["Eat protein within 30 mins after workout", "Sleep 8 hours for muscle recovery",
                       "Hydrate — drink 3L water daily", "Consistency beats intensity every time",
                       "Compound exercises burn more calories", "Rest days are part of the program",
                       "Track your progress to stay motivated", "Warm up before every session"],
        "motivation": ["'The secret is to get started.' — Mark Twain", "Take one small step today",
                       "Failure is just feedback in disguise", "Your future self is watching you now",
                       "'Do it afraid.' — Brené Brown", "Discipline outlasts motivation every time",
                       "Progress beats perfection every single day", "Your environment shapes your mindset"],
        "technology": ["AI processes data 10x faster than humans", "Quantum computers solve in seconds, not years",
                       "5G is 100x faster than 4G networks", "Cybersecurity attacks happen every 39 seconds",
                       "90% of world's data created in last 2 years", "Edge computing reduces latency to milliseconds",
                       "Open-source code powers 90% of the internet", "Blockchain removes the need for middlemen"],
        "finance":    ["Pay yourself first — save before spending", "Invest early, even small amounts compound",
                       "Avoid lifestyle inflation as income grows", "3-6 months emergency fund is essential",
                       "Index funds outperform 80% of active managers", "Automate savings to remove temptation",
                       "Debt with high interest costs you freedom", "Tax-advantaged accounts compound faster"],
        "crypto":     ["Never invest more than you can lose", "Store crypto in cold wallets for safety",
                       "Bitcoin halving happens every 4 years", "Research before you invest — DYOR always",
                       "Diversify across multiple blockchains", "Gas fees vary by time of day",
                       "Smart contracts automate agreements trustlessly", "Stablecoins reduce volatility risk"],
        "marketing":  ["Hook viewers in the first 3 seconds", "Consistency beats viral content long-term",
                       "Email marketing ROI is $42 per $1 spent", "Social proof drives 70% of buying decisions",
                       "Short-form video reaches 3× more people", "User-generated content costs less, converts more",
                       "SEO compounds — write once, rank forever", "Niche audiences convert better than broad ones"],
        "food":       ["Eat the rainbow — variety equals nutrition", "Cooking at home saves 50% vs eating out",
                       "Fermented foods boost gut health naturally", "Dark chocolate has more antioxidants than blueberries",
                       "Olive oil is healthier than vegetable oil", "Meal prepping saves 5 hours a week",
                       "Frozen vegetables retain more nutrients than fresh", "Eat slowly — it takes 20 min to feel full"],
        "travel":     ["Book flights Tuesday for cheapest fares", "Travel insurance costs 5% of trip price",
                       "Local SIM cards save 80% on roaming", "Pack half the clothes, twice the money",
                       "Shoulder season offers 40% cheaper hotels", "Always keep a backup copy of your passport",
                       "Airport lounges can be accessed for $35/day", "Learn 10 local phrases — locals respect it"],
    }
    return fallbacks.get(niche.lower(), [
        "Stay informed and keep learning",
        "Small steps lead to big results",
        "Consistency is the key to success",
        "Knowledge is your greatest asset",
        "Every expert was once a beginner",
        "Focus on progress, not perfection",
        "Your habits shape your future daily",
        "Invest in yourself — it always pays",
    ])
