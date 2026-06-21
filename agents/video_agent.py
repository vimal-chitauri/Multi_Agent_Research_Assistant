import os
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dotenv import load_dotenv
from agents.base_agent import BaseAgent, AgentResult
from tools.video_tools import create_video, create_news_video

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
        self.duration = duration
        self.niche    = niche

    @property
    def system_prompt(self) -> str:
        return f"You are an expert content creator for {self.niche} social media videos. Always respond in valid JSON."

    def run(self, input_data: dict) -> AgentResult:
        posts = input_data.get("posts", [])
        if not posts:
            return self._failure("No posts received.")

        niche = input_data.get("niche", self.niche)

        self._log(
            f"Creating [bold]{self.duration}s[/bold] 9:16 videos | "
            f"niche: [cyan]{niche}[/cyan] | "
            f"posts: {len(posts)} | parallel: up to 3"
        )

        videos_created = 0
        results: list[dict] = []

        def _render(post: dict) -> dict:
            topic     = post["topic"]
            rank      = post.get("rank", 0)
            timestamp = int(time.time() * 1000) + rank
            video_path = str(VIDEOS_DIR / f"post_{rank}_{timestamp}.mp4")

            music_path  = post.get("music_path")
            music_title = post.get("music_title", "No music")
            self._log(
                f"  Rendering: [cyan]{topic[:45]}[/cyan]  |  "
                f"music: {'✓' if music_path else '–'}"
            )

            try:
                if niche.lower() == "latest_news":
                    image_paths = post.get("image_paths") or (
                        [post["image_path"]] if post.get("image_path") else []
                    )
                    if not image_paths:
                        self._log("  [yellow]No images — gradient fallback slides[/yellow]")

                    bullets = post.get("bullets") or self._generate_info_content(topic, niche)

                    create_news_video(
                        image_paths  = image_paths,
                        heading      = post.get("heading", topic),
                        bullets      = bullets,
                        source_name  = post.get("source_name", ""),
                        music_path   = music_path,
                        output_path  = video_path,
                        rank         = rank,
                        voice_path   = post.get("voice_path"),
                        voice_timing = post.get("voice_timing"),
                        hook         = self._generate_hook(topic, niche),
                    )
                else:
                    # Use all available image variants for visual variety
                    image_paths = post.get("image_paths") or (
                        [post["image_path"]] if post.get("image_path") else []
                    )
                    if not image_paths:
                        self._log(f"  [yellow]No image for rank {rank} — skipping[/yellow]")
                        post.update({"video_path": None, "video_status": "skipped — no image"})
                        return post

                    info_lines = self._generate_info_content(topic, niche)

                    create_video(
                        image_paths = image_paths,
                        title       = topic,
                        info_lines  = info_lines,
                        music_path  = music_path,
                        output_path = video_path,
                        duration    = self.duration,
                        niche       = niche,
                        voice_path  = post.get("voice_path"),
                        rank        = rank,
                        hook        = self._generate_hook(topic, niche),
                        watermark   = niche.replace("_", " ").title(),
                    )
                    post["info_content"] = info_lines

                post.update({
                    "video_path":   video_path,
                    "video_status": "created",
                    "music_title":  music_title,
                })
                self._log(f"  [green]✓ {Path(video_path).name}[/green]")

            except Exception as e:
                self._log(f"  [red]Render failed ({topic[:30]}): {e}[/red]")
                post.update({"video_path": None, "video_status": f"failed — {str(e)[:80]}"})

            return post

        with ThreadPoolExecutor(max_workers=min(len(posts), 3)) as pool:
            futures = {pool.submit(_render, post): post for post in posts}
            for fut in as_completed(futures):
                try:
                    result = fut.result()
                    results.append(result)
                    if result.get("video_status") == "created":
                        videos_created += 1
                except Exception as e:
                    orig = futures[fut]
                    orig.update({"video_path": None, "video_status": f"failed — {str(e)[:80]}"})
                    results.append(orig)

        # Restore original ordering by rank
        results.sort(key=lambda p: p.get("rank", 0))

        self._log(f"[green]{videos_created}/{len(posts)} videos created[/green]")
        return self._success(
            data={
                "posts":          results,
                "total":          len(posts),
                "videos_created": videos_created,
                "duration_sec":   self.duration,
            },
            reasoning=f"{videos_created}/{len(posts)} × {self.duration}s 9:16 videos | Ken Burns | niche: {niche}",
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
            hook  = self.think(prompt).strip().strip('"').strip("'")
            words = hook.split()[:12]
            mid   = len(words) // 2
            return " ".join(words[:mid]) + "\n" + " ".join(words[mid:])
        except Exception:
            return None


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
