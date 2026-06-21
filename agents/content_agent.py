import os
import json
from dotenv import load_dotenv
from agents.base_agent import BaseAgent, AgentResult

load_dotenv()


def _get_rag_context(topic: str, niche: str) -> str:
    try:
        from tools.rag_tools import build_content_context
        ctx, _ = build_content_context(topic, niche)
        return ctx
    except Exception:
        return ""


class ContentAgent(BaseAgent):

    def __init__(self, niche: str = None, brand_tone: str = "professional yet engaging"):
        super().__init__(name="ContentAgent", model="smart")
        self.niche = niche or os.getenv("NICHE", "technology")
        self.brand_tone = brand_tone

    @property
    def system_prompt(self) -> str:
        return f"""You are an expert social media content creator specializing in {self.niche}.

Your writing style is: {self.brand_tone}

You create high-performing content for two platforms:

TWITTER rules:
- Max 280 characters (strict)
- Hook in the first line
- Use 1-2 relevant emojis max
- End with 1-2 hashtags only
- Conversational, punchy

INSTAGRAM rules:
- 150-300 words
- Start with a strong hook line
- Tell a mini-story or give real value
- Use line breaks for readability
- End with a call to action
- 10-15 relevant hashtags at the bottom

Always respond in valid JSON only. No extra text."""

    def run(self, input_data: dict) -> AgentResult:
        trends = input_data.get("trends", [])
        if not trends:
            return self._failure("No trends provided. Run TrendAgent first.")

        niche = input_data.get("niche", self.niche)
        self._log(f"Generating content for [bold]{len(trends)}[/bold] trending topics...")

        all_posts = []

        for trend in trends:
            self._log(f"Writing content for: [cyan]{trend['topic']}[/cyan]")
            rag_context = _get_rag_context(trend["topic"], niche)

            if niche.lower() == "latest_news":
                parsed = self._generate_news_content(trend, rag_context)
            else:
                parsed = self._generate_standard_content(trend, niche, rag_context)

            if parsed:
                all_posts.append(parsed)

        if not all_posts:
            return self._failure("Failed to generate content for any trend.")

        self._log(f"[green]Generated {len(all_posts)} posts successfully[/green]")

        return self._success(
            data={"posts": all_posts, "total": len(all_posts)},
            reasoning=f"Generated Twitter + Instagram content for top {len(all_posts)} trending topics in {niche}"
        )

    def _generate_news_content(self, trend: dict, rag_context: str = "") -> dict | None:
        source_url  = trend.get("source_url", "")
        source_name = trend.get("source_name", "")
        full_title  = trend.get("full_title", trend["topic"])
        description = trend.get("description", "")

        rag_section = (f"EXPERT CONTENT GUIDANCE — apply these viral content principles:\n"
                       f"{rag_context}\n"
                       f"---\n") if rag_context else ""

        prompt = f"""
{rag_section}You are a news content creator for social media. Create content for this news story.

NEWS TITLE  : {full_title}
DESCRIPTION : {description}
SOURCE      : {source_name}
URL         : {source_url}
ANGLE       : {trend.get("content_angle", "")}

Generate content in this EXACT JSON format (no other text):
{{
  "topic": "{trend['topic']}",
  "rank": {trend['rank']},
  "heading": "A punchy bold headline under 10 words (all caps style)",
  "bullets": [
    "Key fact 1 — one clear, informative sentence of around 15 words",
    "Key fact 2 — one clear, informative sentence of around 15 words",
    "Key fact 3 — one clear, informative sentence of around 15 words",
    "Key fact 4 — one clear, informative sentence of around 15 words",
    "Key fact 5 — one clear, informative sentence of around 15 words",
    "Key fact 6 — one clear, informative sentence of around 15 words",
    "Key fact 7 — one clear, informative sentence of around 15 words",
    "Key fact 8 — one clear, informative sentence of around 15 words",
    "Key fact 9 — one clear, informative sentence of around 15 words",
    "Key fact 10 — one clear, informative sentence of around 15 words"
  ],
  "description": "180-200 words describing this news story for Instagram. Engaging, factual, covers background context and impact, ends with a question or CTA.",
  "source_url": "{source_url}",
  "source_name": "{source_name}",
  "hashtags": ["news", "breaking", "trending", "world", "latest", "up to 10 relevant hashtags"],
  "image_prompts": [
    "Cinematic scene for slide 1 — symbolic or environmental visual representing this news topic. NO real people, NO named individuals, NO faces. Use objects, flags, buildings, skylines, maps, symbols, nature. Professional photography, dramatic lighting.",
    "Scene for slide 2 — a different symbolic angle of the same topic. NO real people or faces. Wide shot, editorial photography style, cinematic composition.",
    "Scene for slide 3 — hopeful or impactful symbolic conclusion visual. NO real people or faces. Cinematic 4K, dramatic lighting."
  ],
  "instagram": {{
    "caption": "description text + \\n\\nSource: {source_name}\\n{source_url}",
    "hashtags": ["news", "breaking", "world", "trending"]
  }}
}}
"""
        response = self.think(prompt)
        parsed = self._parse_response(response, trend["topic"])
        if parsed:
            parsed.setdefault("source_url", source_url)
            parsed.setdefault("source_name", source_name)
        return parsed

    def _generate_standard_content(self, trend: dict, niche: str, rag_context: str = "") -> dict | None:
        rag_section = (f"EXPERT CONTENT GUIDANCE — apply these viral content principles:\n"
                       f"{rag_context}\n"
                       f"---\n") if rag_context else ""

        prompt = f"""
{rag_section}Create social media content for this trending topic:

TOPIC       : {trend['topic']}
WHY TRENDING: {trend['why_trending']}
EMOTION     : {trend['emotion']}
POST TYPE   : {trend['post_type']}
AUDIENCE    : {trend['target_audience']}
ANGLE       : {trend['content_angle']}
NICHE       : {niche}

Generate content in this EXACT JSON format (no other text):
{{
  "topic": "{trend['topic']}",
  "rank": {trend['rank']},
  "twitter": {{
    "text": "the tweet text under 280 chars including hashtags",
    "char_count": 0,
    "hashtags": ["hashtag1", "hashtag2"]
  }},
  "instagram": {{
    "caption": "full instagram caption with line breaks using \\n",
    "hook": "the first attention-grabbing line",
    "hashtags": ["hashtag1", "hashtag2", "...up to 15 hashtags"],
    "call_to_action": "the closing CTA line"
  }},
  "image_prompt": "detailed description for AI image generation",
  "best_post_time": "e.g. Tuesday 6-8pm EST",
  "why_this_works": "brief explanation"
}}
"""
        parsed = self._parse_response(self.think(prompt), trend["topic"])
        if parsed:
            parsed["twitter"]["char_count"] = len(parsed["twitter"]["text"])
        return parsed

    def _parse_response(self, response: str, topic: str) -> dict | None:
        try:
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            return json.loads(cleaned.strip())
        except json.JSONDecodeError:
            self._log(f"[red]Failed to parse JSON for topic: {topic}[/red]")
            return None
