import os
import time
from dotenv import load_dotenv
from agents.base_agent import BaseAgent, AgentResult
from tools.publisher_tools import (
    facebook_post_video,
    instagram_create_reel,
    instagram_wait_until_ready,
    instagram_publish_reel,
)
from tools.storage_tools import upload_to_b2, delete_from_b2

load_dotenv(override=True)


def _ig_error_hint(error: str) -> str:
    e = error.lower()
    if "190" in error or "invalid oauth" in e or "access token" in e:
        return ("\n  → Your INSTAGRAM_ACCESS_TOKEN is expired or invalid. "
                "Generate a new long-lived token in Meta Business Suite → "
                "Settings → Advanced → Instagram Graph API.")
    if "10" in error and "permission" in e:
        return ("\n  → Missing 'instagram_content_publish' permission. "
                "Re-authorise your app in Meta App Review and request this permission.")
    if "9007" in error or "not available" in e:
        return ("\n  → Instagram could not fetch the video URL from Cloudinary. "
                "Check that your Cloudinary account is active and the video URL is publicly reachable.")
    if "2207026" in error or "invalid video" in e:
        return ("\n  → Video format rejected. Instagram Reels require: "
                "MP4 / MOV, H.264 codec, AAC audio, 9:16 aspect ratio, 500×888 px minimum, under 1 GB.")
    if "2207051" in error or "too short" in e:
        return "\n  → Reel is too short. Minimum duration is 3 seconds."
    if "2207006" in error or "aspect ratio" in e:
        return "\n  → Wrong aspect ratio. Instagram Reels must be 9:16 (vertical). Check video_tools.py output size."
    if "account_id" in e or "account" in e:
        return ("\n  → INSTAGRAM_BUSINESS_ACCOUNT_ID may be wrong. "
                "Confirm it is your Instagram Business/Creator account numeric ID, not the Facebook Page ID.")
    return ""


class PublisherAgent(BaseAgent):

    def __init__(
        self,
        dry_run: bool = True,
        publish_facebook: bool = True,
        publish_instagram: bool = True,
    ):
        super().__init__(name="PublisherAgent", model="fast")
        self.dry_run           = dry_run
        self.pub_facebook      = publish_facebook
        self.pub_instagram     = publish_instagram

        self.fb_page_id    = os.getenv("FACEBOOK_PAGE_ID")
        self.fb_page_token = os.getenv("FACEBOOK_PAGE_TOKEN")

        self.ig_token      = os.getenv("INSTAGRAM_ACCESS_TOKEN")
        self.ig_account_id = os.getenv("INSTAGRAM_BUSINESS_ACCOUNT_ID")

    @property
    def system_prompt(self) -> str:
        return "You are a social media publishing assistant. Be concise and factual."

    def run(self, input_data: dict) -> AgentResult:
        posts = input_data.get("posts", [])
        if not posts:
            return self._failure("No approved posts to publish.")

        mode = "[DRY RUN]" if self.dry_run else "[LIVE]"
        platforms = []
        if self.pub_facebook:  platforms.append("Facebook")
        if self.pub_instagram: platforms.append("Instagram")

        self._log(f"{mode} Publishing {len(posts)} post(s) to: {', '.join(platforms)}")

        if not self.dry_run:
            if self.pub_facebook and (not self.fb_page_id or not self.fb_page_token):
                return self._failure("Missing FACEBOOK_PAGE_ID or FACEBOOK_PAGE_TOKEN in .env")
            if self.pub_instagram and (not self.ig_token or not self.ig_account_id):
                return self._failure("Missing INSTAGRAM_ACCESS_TOKEN or INSTAGRAM_BUSINESS_ACCOUNT_ID in .env")
            if self.pub_instagram and not os.getenv("CLOUDINARY_CLOUD_NAME"):
                return self._failure("Missing Cloudinary credentials (CLOUDINARY_CLOUD_NAME etc.) in .env")

        published = []
        failed    = []

        for post in posts:
            self._log(f"\nProcessing: [cyan]{post['topic']}[/cyan]")
            result = self._publish_post(post)

            if result.get("any_success"):
                published.append(result)
            else:
                failed.append(result)

            if not self.dry_run and len(posts) > 1:
                time.sleep(5)

        self._log(f"\n[green]Done: {len(published)} published, {len(failed)} failed[/green]")

        return self._success(
            data={
                "published": published,
                "failed":    failed,
                "dry_run":   self.dry_run,
                "total":     len(published),
                "platforms": platforms,
            },
            reasoning=f"{len(published)}/{len(posts)} posts | {', '.join(platforms)}"
        )

    def _publish_post(self, post: dict) -> dict:
        topic      = post["topic"]
        caption    = self._build_caption(post)
        video_path = post.get("video_path")
        score      = post.get("score", "N/A")

        result = {
            "topic":      topic,
            "caption":    caption,
            "score":      score,
            "facebook":   None,
            "instagram":  None,
            "any_success": False,
        }

        if self.dry_run:
            self._show_dry_run_preview(post, caption, video_path)
            result["facebook"]    = {"success": True, "url": "https://facebook.com [DRY RUN]", "platform": "facebook", "post_id": "dry_run"}
            result["instagram"]   = {"success": True, "url": "https://instagram.com [DRY RUN]", "platform": "instagram", "post_id": "dry_run"}
            result["any_success"] = True
            self._index_to_rag(post, result["instagram"])
            return result

        if not video_path or not os.path.exists(video_path):
            self._log("  [red]Video file missing — skipping[/red]")
            result["facebook"]  = {"success": False, "error": "No video file", "platform": "facebook"}
            result["instagram"] = {"success": False, "error": "No video file", "platform": "instagram"}
            return result

        if self.pub_facebook:
            self._log("  [bold]── Facebook ──[/bold]")
            fb_result = facebook_post_video(
                video_path  = video_path,
                caption     = caption,
                page_id     = self.fb_page_id,
                page_token  = self.fb_page_token,
            )
            result["facebook"] = fb_result
            if fb_result["success"]:
                result["any_success"] = True
                self._log(f"  [green]Facebook: {fb_result['url']}[/green]")
                self._index_to_rag(post, fb_result)
            else:
                self._log(f"  [red]Facebook: {fb_result.get('error')}[/red]")

        if self.pub_instagram:
            self._log("  [bold]── Instagram ──[/bold]")
            ig_result = self._post_to_instagram(video_path, caption)
            result["instagram"] = ig_result
            if ig_result["success"]:
                result["any_success"] = True
                self._log(f"  [green]Instagram: {ig_result['url']}[/green]")
                self._index_to_rag(post, ig_result)
            else:
                self._log(f"  [red]Instagram: {ig_result.get('error')}[/red]")

        if result["any_success"]:
            self._cleanup_local_files(post)

        return result

    def _index_to_rag(self, post: dict, result: dict):
        try:
            from tools.rag_tools import index_published_post
            ok = index_published_post(post, result)
            if ok:
                self._log("  [dim]RAG: post indexed for future content improvement[/dim]")
        except Exception as e:
            self._log(f"  [dim]RAG index skipped: {e}[/dim]")

    def _cleanup_local_files(self, post: dict):
        for key, label in [("video_path", "video"), ("image_path", "image")]:
            path = post.get(key)
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                    self._log(f"  Deleted local {label}: {os.path.basename(path)}")
                except Exception as e:
                    self._log(f"  [yellow]Could not delete {label}: {e}[/yellow]")

    def _post_to_instagram(self, video_path: str, caption: str) -> dict:
        b2_url = b2_file_id = b2_file_name = None

        try:
            self._log("  Step 1/4: Uploading to Cloudinary...")
            b2_url, b2_file_id, b2_file_name = upload_to_b2(video_path)
            if not b2_url:
                return {
                    "success": False,
                    "error":   "Cloudinary upload failed — check CLOUDINARY_CLOUD_NAME / API_KEY / API_SECRET in .env",
                    "platform": "instagram",
                }

            self._log("  Step 2/4: Creating Instagram container...")
            self._log(f"  Video URL: {b2_url[:80]}…")
            container_id, ig_error = instagram_create_reel(
                video_url    = b2_url,
                caption      = caption,
                account_id   = self.ig_account_id,
                access_token = self.ig_token,
            )
            if not container_id:
                hint = _ig_error_hint(ig_error or "")
                return {
                    "success":  False,
                    "error":    f"Instagram container creation failed: {ig_error}{hint}",
                    "platform": "instagram",
                }

            self._log("  Step 3/4: Waiting for Instagram processing...")
            ready = instagram_wait_until_ready(container_id, self.ig_token, max_secs=180)
            if not ready:
                return {"success": False, "error": "Instagram video processing timed out after 3 min", "platform": "instagram"}

            self._log("  Step 4/4: Publishing Reel...")
            result = instagram_publish_reel(container_id, self.ig_account_id, self.ig_token)
            if result:
                post_id, permalink = result
                return {
                    "success":  True,
                    "post_id":  post_id,
                    "url":      permalink,
                    "platform": "instagram",
                }
            return {"success": False, "error": "Publish step returned no post_id", "platform": "instagram"}

        finally:
            if b2_file_id and b2_file_name:
                self._log("  Cleaning up Cloudinary temp file...")
                delete_from_b2(b2_file_id, b2_file_name)

    def _build_caption(self, post: dict) -> str:
        if "description" in post and post.get("description"):
            caption  = post["description"]
            source   = post.get("source_url", "")
            if source:
                caption += f"\n\nSource: {post.get('source_name', '')}\n{source}"
            hashtags = post.get("hashtags", [])
        else:
            caption  = post["instagram"]["caption"]
            hashtags = post["instagram"]["hashtags"]
        clean = [f"#{t.strip().lstrip('#')}" for t in hashtags if t.strip()]
        return f"{caption}\n\n{' '.join(clean)}"

    def _show_dry_run_preview(self, post: dict, caption: str, video_path: str):
        sep = "─" * 58
        print(f"\n{sep}")
        print(f"  DRY RUN — What would be posted")
        print(sep)
        print(f"  TOPIC     : {post['topic']}")
        print(f"  SCORE     : {post.get('score', 'N/A')}/100")
        print(f"  VIDEO     : {video_path or 'No video'}")
        print(f"\n  CAPTION   :\n  {caption[:300].replace(chr(10), chr(10) + '  ')}")
        if self.pub_facebook:
            print(f"\n  FACEBOOK  : Would upload to page {self.fb_page_id or '[not set]'}")
        if self.pub_instagram:
            print(f"  INSTAGRAM : Would post as Reel to account {self.ig_account_id or '[not set]'}")
            print(f"              (via Cloudinary temp host)")
        print(f"{sep}\n")
