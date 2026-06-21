import json
import os
from agents.base_agent import BaseAgent, AgentResult
from dotenv import load_dotenv

load_dotenv()

QUALITY_THRESHOLD = 70


class VerifierAgent(BaseAgent):

    def __init__(self, niche: str = None, ui_mode: bool = False):
        super().__init__(name="VerifierAgent", model="fast")
        self.niche   = niche or os.getenv("NICHE", "technology")
        self.ui_mode = ui_mode

    @property
    def system_prompt(self) -> str:
        return f"""You are a strict social media content reviewer for a {self.niche} brand.

Your job is to evaluate posts BEFORE they are published.

Score each post on these 5 dimensions (0-20 each, total 0-100):
1. ACCURACY     — Is the information factually correct and not misleading?
2. TONE         — Does it match the brand voice? Professional yet engaging?
3. PLATFORM FIT — Does Twitter post fit 280 chars? Is Instagram caption rich enough?
4. ENGAGEMENT   — Will it get likes, comments, shares? Is the hook strong?
5. SAFETY       — No controversial, harmful, or off-brand content?

Be strict. A score below 70 means the post needs revision.
Always respond in valid JSON only."""

    def run(self, input_data: dict) -> AgentResult:
        posts = input_data.get("posts", [])
        if not posts:
            return self._failure("No posts to verify. Run ContentAgent first.")

        self._log(f"Reviewing [bold]{len(posts)}[/bold] posts...")

        reviewed_posts = []

        for post in posts:
            self._log(f"Reviewing: [cyan]{post['topic']}[/cyan]")

            review = self._llm_review(post)
            post["review"] = review

            score = review.get("total_score", 0)
            status = "PASS" if score >= QUALITY_THRESHOLD else "FAIL"
            color = "green" if status == "PASS" else "red"

            self._log(
                f"  Score: [{color}]{score}/100[/{color}]  "
                f"Status: [{color}]{status}[/{color}]"
            )

            if score < QUALITY_THRESHOLD:
                self._log(f"  [yellow]Issues: {review.get('issues', [])}[/yellow]")

            reviewed_posts.append({**post, "llm_status": status, "score": score})

        if self.ui_mode:
            approved_posts = reviewed_posts  # all pass through; UI decides
        else:
            approved_posts = self._human_approval_gate(reviewed_posts)

        if not approved_posts:
            return self._failure("No posts approved by human reviewer.")

        self._log(f"[green]{len(approved_posts)} post(s) approved and ready to publish[/green]")

        return self._success(
            data={"posts": approved_posts, "total": len(approved_posts)},
            reasoning=f"{len(approved_posts)}/{len(posts)} posts approved after LLM + human review"
        )

    def _llm_review(self, post: dict) -> dict:
        prompt = f"""
Review this social media post and score it strictly:

TOPIC: {post['topic']}

CONTENT:
{post['heading'] + chr(10) + chr(10).join(post.get('bullets', [])) if 'heading' in post else post.get('twitter', {}).get('text', '')}

DESCRIPTION / CAPTION:
{post.get('description') or post.get('instagram', {}).get('caption', '')}

HASHTAGS: {post.get('hashtags') or post.get('instagram', {}).get('hashtags', [])}

Respond ONLY with this JSON (no other text):
{{
  "scores": {{
    "accuracy":     0,
    "tone":         0,
    "platform_fit": 0,
    "engagement":   0,
    "safety":       0
  }},
  "total_score": 0,
  "strengths": ["what works well"],
  "issues": ["what needs improvement"],
  "verdict": "PASS or FAIL",
  "suggestion": "one specific improvement if score < 70, else empty string"
}}
"""
        response = self.think(prompt)

        try:
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            result = json.loads(cleaned.strip())
            scores = result.get("scores", {})
            result["total_score"] = sum(scores.values())
            return result
        except json.JSONDecodeError:
            return {
                "scores": {},
                "total_score": 50,
                "strengths": [],
                "issues": ["Could not parse LLM review"],
                "verdict": "PASS",
                "suggestion": "",
            }

    def _human_approval_gate(self, posts: list[dict]) -> list[dict]:
        approved = []

        print("\n" + "=" * 60)
        print("  HUMAN APPROVAL GATE")
        print("  Review each post and decide: approve or skip")
        print("=" * 60)

        for post in posts:
            score = post["score"]
            status = post["llm_status"]
            review = post.get("review", {})

            print(f"\n{'─' * 60}")
            print(f"POST #{post['rank']}  |  Topic: {post['topic']}")
            print(f"LLM Score: {score}/100  |  Status: {status}")
            print(f"{'─' * 60}")

            if "heading" in post:
                print(f"\n[HEADING] {post['heading']}")
                for b in post.get("bullets", []):
                    print(f"  • {b}")
                print(f"\n[DESCRIPTION]\n{post.get('description','')[:200]}")
                print(f"\n[SOURCE] {post.get('source_url','')}")
            else:
                print(f"\n[TWITTER - {post.get('twitter',{}).get('char_count',0)} chars]")
                print(post.get("twitter", {}).get("text", ""))
                print(f"\n[INSTAGRAM HOOK]")
                print(post.get("instagram", {}).get("hook", ""))
                print(f"\n[INSTAGRAM CAPTION]")
                print(post.get("instagram", {}).get("caption", ""))
                print(f"\n[IMAGE PROMPT]")
                print(post.get("image_prompt", ""))

            print(f"\n[LLM FEEDBACK]")
            print(f"  Strengths : {review.get('strengths', [])}")
            print(f"  Issues    : {review.get('issues', [])}")
            if review.get("suggestion"):
                print(f"  Suggestion: {review.get('suggestion')}")

            print(f"\n[BEST TIME TO POST] {post['best_post_time']}")

            while True:
                choice = input(
                    f"\n  Approve this post? [y = yes / n = skip / q = quit all]: "
                ).strip().lower()

                if choice == "y":
                    approved.append(post)
                    print(f"  ✓ Post #{post['rank']} approved.")
                    break
                elif choice == "n":
                    print(f"  ✗ Post #{post['rank']} skipped.")
                    break
                elif choice == "q":
                    print("  Stopping review.")
                    return approved
                else:
                    print("  Please enter y, n, or q.")

        return approved
