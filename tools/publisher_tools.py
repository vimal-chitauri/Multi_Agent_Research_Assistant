import os
import time
import requests
from dotenv import load_dotenv

load_dotenv(override=True)

GRAPH_VERSION = "v19.0"
GRAPH_URL     = f"https://graph.facebook.com/{GRAPH_VERSION}"
GRAPH_VIDEO   = f"https://graph-video.facebook.com/{GRAPH_VERSION}"



def facebook_post_video(
    video_path: str,
    caption: str,
    page_id: str,
    page_token: str,
) -> dict:
    url = f"{GRAPH_VIDEO}/{page_id}/videos"
    print("  Facebook: uploading video file...")

    try:
        with open(video_path, "rb") as f:
            r = requests.post(
                url,
                data={
                    "description":  caption,
                    "access_token": page_token,
                },
                files={"source": ("video.mp4", f, "video/mp4")},
                timeout=180,
            )

        if r.status_code == 200:
            video_id = r.json().get("id")
            post_url = f"https://www.facebook.com/{page_id}/videos/{video_id}"
            print(f"  Facebook: posted — {post_url}")
            return {"success": True, "post_id": video_id, "url": post_url, "platform": "facebook"}

        error = r.json().get("error", {}).get("message", r.text[:120])
        print(f"  Facebook: upload failed — {error}")
        return {"success": False, "error": error, "platform": "facebook"}

    except FileNotFoundError:
        return {"success": False, "error": "Video file not found", "platform": "facebook"}
    except Exception as e:
        print(f"  Facebook: error — {e}")
        return {"success": False, "error": str(e), "platform": "facebook"}



def instagram_create_reel(
    video_url: str,
    caption: str,
    account_id: str,
    access_token: str,
) -> tuple[str | None, str | None]:
    print("  Instagram: creating Reels container...")
    try:
        r = requests.post(
            f"{GRAPH_URL}/{account_id}/media",
            data={
                "media_type":    "REELS",
                "video_url":     video_url,
                "caption":       caption,
                "share_to_feed": "true",
                "access_token":  access_token,
            },
            timeout=30,
        )
        data = r.json()
        if r.status_code == 200:
            cid = data.get("id")
            print(f"  Instagram: container created ({cid})")
            return cid, None
        err_obj  = data.get("error", {})
        err_msg  = err_obj.get("message", r.text[:200])
        err_code = err_obj.get("code", "")
        err_sub  = err_obj.get("error_subcode", "")
        full_err = f"[{err_code}/{err_sub}] {err_msg}" if err_code else err_msg
        print(f"  Instagram: container failed — {full_err}")
        return None, full_err
    except Exception as e:
        msg = str(e)
        print(f"  Instagram: container error — {msg}")
        return None, msg


def instagram_wait_until_ready(
    container_id: str,
    access_token: str,
    max_secs: int = 180,
) -> bool:
    print(f"  Instagram: waiting for processing (max {max_secs}s)...")
    waited = 0
    while waited < max_secs:
        try:
            r = requests.get(
                f"{GRAPH_URL}/{container_id}",
                params={"fields": "status_code", "access_token": access_token},
                timeout=15,
            )
            status = r.json().get("status_code", "UNKNOWN")
            print(f"    [{waited}s] status: {status}")
            if status == "FINISHED":
                return True
            if status == "ERROR":
                print("  Instagram: processing ERROR")
                return False
        except Exception as e:
            print(f"  Instagram: poll error — {e}")
        time.sleep(8)
        waited += 8
    print(f"  Instagram: timed out after {max_secs}s")
    return False


def instagram_publish_reel(
    container_id: str,
    account_id: str,
    access_token: str,
) -> str | None:
    print("  Instagram: publishing Reel...")
    try:
        r = requests.post(
            f"{GRAPH_URL}/{account_id}/media_publish",
            data={
                "creation_id":  container_id,
                "access_token": access_token,
            },
            timeout=30,
        )
        if r.status_code == 200:
            post_id = r.json().get("id")
            pr = requests.get(f"{GRAPH_URL}/{post_id}", params={
                "fields": "permalink", "access_token": access_token
            }, timeout=15)
            permalink = pr.json().get("permalink", f"https://www.instagram.com/")
            print(f"  Instagram: live! {permalink}")
            return post_id, permalink
        error = r.json().get("error", {}).get("message", r.text[:120])
        print(f"  Instagram: publish failed — {error}")
    except Exception as e:
        print(f"  Instagram: publish error — {e}")
    return None, None
