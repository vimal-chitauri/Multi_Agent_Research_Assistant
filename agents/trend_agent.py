import os
import json
from dotenv import load_dotenv
from agents.base_agent import BaseAgent, AgentResult
from tools.trend_tools import (
    get_scored_trends,
    fetch_all_signals,
    get_multi_category_headlines,
    get_top_headlines,
    score_topics,
)

load_dotenv()


class TrendAgent(BaseAgent):

    def __init__(self, niche: str = None):
        super().__init__(name="TrendAgent", model="smart")
        self.niche = niche or os.getenv("NICHE", "technology")

    @property
    def system_prompt(self) -> str:
        return (
            f"You are a social media trend analyst specializing in {self.niche} content. "
            f"You receive pre-scored trending topics and make the final editorial decision "
            f"on which topics will perform best for short-form video. "
            f"Always respond in valid JSON."
        )

    def run(self, input_data: dict = None) -> AgentResult:
        niche   = (input_data or {}).get("niche", self.niche)
        exclude = (input_data or {}).get("exclude_topics", [])
        count   = (input_data or {}).get("count", 3)

        self._log(f"Discovering trends | niche: [bold]{niche}[/bold] | want: {count} topics")

        if niche.lower() == "latest_news":
            return self._run_latest_news(exclude_topics=exclude, count=count)

        self._log("Fetching signals in parallel: Google · YouTube · Reddit · NewsAPI · RSS …")
        signals = fetch_all_signals(niche, parallel=True)

        news    = [a for a in signals.get("news",    []) if "error" not in a]
        youtube = signals.get("youtube", [])
        rss     = signals.get("rss",     [])
        twitter = signals.get("twitter", [])
        self._log(
            f"  Fetched in {signals.get('fetch_ms', '?')}ms  |  "
            f"Google daily={len(signals.get('google_daily', []))}  "
            f"rising={len(signals.get('google_rising', []))}  "
            f"Reddit={len(signals.get('reddit', []))}  "
            f"News={len(news)}  "
            f"YouTube={len(youtube)}  "
            f"RSS={len(rss)}  "
            f"Twitter={len(twitter)}"
        )

        scored = score_topics(
            google_daily   = signals.get("google_daily",  []),
            google_rising  = signals.get("google_rising", []),
            reddit_posts   = signals.get("reddit",        []),
            news_articles  = signals.get("news",          []),
            hackernews     = signals.get("hackernews",    []),
            youtube_videos = youtube,
            twitter_topics = twitter,
            rss_articles   = rss,
        )

        if not scored:
            self._log("[yellow]No scored topics — falling back to HackerNews signal[/yellow]")
            hn = signals.get("hackernews", [])
            scored = [
                {
                    "topic":     s["title"],
                    "score":     min(100.0, s["score"] / 5),
                    "sources":   ["hackernews"],
                    "n_sources": 1,
                }
                for s in hn[:10]
                if not isinstance(s, dict) or "error" not in s
            ]

        if not scored:
            return self._failure("No trend signals retrieved from any source.")

        exclude_keys = {t.lower()[:30] for t in exclude}
        candidates   = [
            t for t in scored
            if t["topic"][:30].lower() not in exclude_keys
        ][:10]

        candidates_txt = "\n".join(
            f"  {i+1:2}. [{t['score']:5.1f}] {t['topic'][:90]}"
            f"  (sources: {', '.join(t['sources'])})"
            for i, t in enumerate(candidates)
        )

        self._log(f"Top candidates by score:\n{candidates_txt}")

        exclusion_block = ""
        if exclude:
            exclusion_block = (
                "\n\nALREADY USED — DO NOT REPEAT:\n"
                + "\n".join(f"  - {t}" for t in exclude)
            )

        prompt = f"""
You are a social media content strategist for {niche}. Below are the top trending topics
today, already scored and ranked by our signal system:
Google 30% + YouTube 20% + Reddit 20% + News 15% + Twitter 10% + RSS 5%
Topics appearing across multiple sources get a cross-source boost (+12 pts each).

YOUR JOB: Pick the best {count} topics from this list for short-form video content.

PREFER topics that:
  ✓ Are broad enough for a global audience
  ✓ Have a clear educational, inspiring, or curiosity angle
  ✓ A compelling 8-bullet script can be written about them
  ✓ Confirmed by multiple sources (google + youtube + rss = very strong signal)
  ✓ NOT just the score — a #5 topic with a great video angle beats a boring #1
{exclusion_block}

PRE-SCORED TRENDING TOPICS (rank · score · topic · sources):
{candidates_txt}

Respond ONLY with this JSON (no other text):
{{
  "trends": [
    {{
      "rank": 1,
      "topic": "exact topic text from the list above",
      "score": <the score number from the list>,
      "why_trending": "one sentence explaining the trend",
      "emotion": "curiosity | inspiration | excitement | humor | wonder",
      "post_type": "informative | entertaining | news | tips | educational",
      "target_audience": "who this appeals to",
      "content_angle": "the specific angle that makes this a great video",
      "sources": ["google", "youtube", "rss"]
    }}
  ],
  "reasoning": "why these {count} topics were chosen over the others"
}}
"""
        self._log("Sending top candidates to LLM for editorial selection …")
        response = self.think(prompt)

        try:
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            parsed = json.loads(cleaned.strip())

            n_found = len(parsed.get("trends", []))
            self._log(f"[bold green]{n_found} topics selected[/bold green]")

            parsed["signal_summary"] = {
                "total_candidates": len(scored),
                "fetch_ms":         signals.get("fetch_ms", 0),
                "sources_active":   [
                    s for s in ["google", "youtube", "reddit", "news", "hackernews", "rss", "twitter"]
                    if any(s in t.get("sources", []) for t in scored[:10])
                ],
            }

            return self._success(data=parsed, reasoning=parsed.get("reasoning", ""))

        except json.JSONDecodeError as e:
            return self._failure(f"LLM returned invalid JSON: {e}\nRaw: {response[:300]}")

    def _run_latest_news(self, exclude_topics: list = None, count: int = 3) -> AgentResult:
        self._log("Fetching multi-category news headlines …")
        articles = get_multi_category_headlines(articles_per_category=5)

        valid = [a for a in articles if "error" not in a and a.get("title") and a.get("url")]
        if not valid:
            self._log("[yellow]Multi-category empty — falling back to top-headlines[/yellow]")
            articles = get_top_headlines(country="us", page_size=15)
            valid = [a for a in articles if "error" not in a and a.get("title") and a.get("url")]

        if not valid:
            return self._failure("No news headlines found. Check NEWS_API_KEY in .env")

        self._log(f"Found {len(valid)} headlines across categories — filtering to top {count} …")

        exclusion_block = ""
        if exclude_topics:
            exclusion_block = (
                "\n\nALREADY USED — DO NOT REPEAT THESE TOPICS:\n"
                + "\n".join(f"  - {t}" for t in exclude_topics)
            )

        prompt = f"""
You are a strict social media news editor. From the articles below, choose EXACTLY {count} most
engaging, shareable, and SAFE stories for Instagram and Facebook family-friendly audiences.
{exclusion_block}

━━━ MANDATORY REJECTION RULES (auto-reject if ANY apply) ━━━
❌  Arrests, criminal charges, murders, kidnappings, domestic violence
❌  Graphic violence, war casualties, death tolls, terrorist attacks
❌  Aggressive partisan political attacks or election controversies
❌  Drug use, addiction, overdose stories
❌  Scandals that name individuals in negative/criminal contexts
❌  Adult or sexually suggestive content
❌  Content that could cause panic or fear

━━━ PREFERRED TOPICS ━━━
✅  Diplomatic wins, international agreements, peacekeeping
✅  Sports records, achievements, inspiring athlete stories
✅  Entertainment, film releases, music, positive celebrity news
✅  Science discoveries, space exploration, health breakthroughs
✅  Business milestones, innovation, startups, economic wins
✅  Inspiring human interest stories, communities helping each other
✅  Technology launches, AI advances, environmental solutions

NEWS ARTICLES ({len(valid)} articles):
{json.dumps(valid, indent=2)}

Return EXACTLY {count} trend(s). Only include what is genuinely safe.

Respond ONLY with this JSON, no other text:
{{
  "trends": [
    {{
      "rank": 1,
      "topic": "short topic name (5-8 words max)",
      "why_trending": "brief explanation",
      "emotion": "curiosity | inspiration | excitement | wonder | pride",
      "post_type": "news",
      "target_audience": "general public",
      "content_angle": "positive/interesting angle that makes this shareable",
      "source_url": "exact URL from article",
      "source_name": "source name from article",
      "full_title": "original article title",
      "description": "article description",
      "category": "category this article came from"
    }}
  ],
  "reasoning": "why these stories were chosen and how they meet safety criteria"
}}
"""
        response = self.think(prompt)
        try:
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            parsed = json.loads(cleaned.strip())
            self._log(f"Selected [bold green]{len(parsed['trends'])}[/bold green] safe news stories")
            return self._success(data=parsed, reasoning=parsed.get("reasoning", ""))
        except json.JSONDecodeError as e:
            return self._failure(f"LLM returned invalid JSON: {e}\nRaw: {response[:300]}")
