import os
import time
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _mask(key: str) -> str:
    if not key or len(key) < 8:
        return "not set"
    return key[:6] + "·····" + key[-4:]


def check_groq() -> dict:
    api_key = os.getenv("GROQ_API_KEY", "")
    base = {
        "id": "groq", "name": "Groq", "type": "LLM Provider",
        "icon": "🧠", "key_preview": _mask(api_key),
        "note": "Free: 100K tokens/day per model (70B), 14.4K req/day (8B)",
        "models": ["llama-3.3-70b-versatile", "llama-4-scout-17b", "llama-3.1-8b-instant"],
        "checked_at": _now(),
    }
    if not api_key:
        return {**base, "status": "not_configured", "status_text": "No API Key",
                "detail": "Add GROQ_API_KEY to .env", "usage_pct": None}
    try:
        t0 = time.time()
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": "llama-3.1-8b-instant",
                  "messages": [{"role": "user", "content": "ping"}],
                  "max_tokens": 1},
            timeout=8,
        )
        latency = int((time.time() - t0) * 1000)

        if resp.status_code == 200:
            h = resp.headers
            limit     = int(h.get("x-ratelimit-limit-tokens", 14400))
            remaining = int(h.get("x-ratelimit-remaining-tokens", limit))
            used      = max(0, limit - remaining)
            pct       = int(used / limit * 100) if limit else 0
            reset_ms  = h.get("x-ratelimit-reset-tokens", "unknown")
            return {**base, "status": "active", "status_text": "Active",
                    "latency_ms": latency,
                    "usage_pct": pct,
                    "usage_text": f"{used:,} / {limit:,} tokens used",
                    "detail": f"Reset: {reset_ms}  ·  Latency: {latency}ms"}

        if resp.status_code == 429:
            data = resp.json().get("error", {})
            return {**base, "status": "rate_limited", "status_text": "Rate Limited",
                    "usage_pct": 100,
                    "usage_text": "Daily token limit reached",
                    "detail": data.get("message", "Try again later")[:120]}

        if resp.status_code == 401:
            return {**base, "status": "error", "status_text": "Invalid Key",
                    "detail": "API key rejected — check GROQ_API_KEY in .env", "usage_pct": None}

        return {**base, "status": "error", "status_text": f"HTTP {resp.status_code}",
                "detail": resp.text[:120], "usage_pct": None}

    except requests.exceptions.Timeout:
        return {**base, "status": "error", "status_text": "Timeout", "detail": "API did not respond in 8s", "usage_pct": None}
    except Exception as e:
        return {**base, "status": "error", "status_text": "Error", "detail": str(e)[:120], "usage_pct": None}


def check_newsapi() -> dict:
    api_key = os.getenv("NEWS_API_KEY", "")
    base = {
        "id": "newsapi", "name": "NewsAPI", "type": "News Data",
        "icon": "📰", "key_preview": _mask(api_key),
        "note": "Free: 100 requests/day · Developer plan",
        "checked_at": _now(),
    }
    if not api_key:
        return {**base, "status": "not_configured", "status_text": "No API Key",
                "detail": "Add NEWS_API_KEY to .env", "usage_pct": None}
    try:
        t0 = time.time()
        resp = requests.get(
            "https://newsapi.org/v2/top-headlines",
            params={"country": "us", "pageSize": 1, "apiKey": api_key},
            timeout=8,
        )
        latency = int((time.time() - t0) * 1000)
        if resp.status_code == 200:
            total = resp.json().get("totalResults", 0)
            return {**base, "status": "active", "status_text": "Active",
                    "latency_ms": latency, "usage_pct": None,
                    "detail": f"Connected · {total:,} articles available · Latency: {latency}ms"}
        if resp.status_code == 429:
            return {**base, "status": "rate_limited", "status_text": "Rate Limited",
                    "usage_pct": 100, "usage_text": "100/100 requests used",
                    "detail": "Daily limit reached — resets at midnight UTC"}
        if resp.status_code == 401:
            return {**base, "status": "error", "status_text": "Invalid Key",
                    "detail": "Check NEWS_API_KEY in .env", "usage_pct": None}
        return {**base, "status": "error", "status_text": f"HTTP {resp.status_code}",
                "detail": resp.json().get("message", "")[:120], "usage_pct": None}
    except Exception as e:
        return {**base, "status": "error", "status_text": "Error", "detail": str(e)[:120], "usage_pct": None}


def check_huggingface() -> dict:
    token = os.getenv("HUGGINGFACE_TOKEN", "")
    base = {
        "id": "huggingface", "name": "HuggingFace", "type": "Image Generation",
        "icon": "🤗", "key_preview": _mask(token),
        "note": "Free tier: FLUX/SDXL require Pro ($9/mo). SD 2.1 may work free.",
        "checked_at": _now(),
    }
    if not token:
        return {**base, "status": "not_configured", "status_text": "No Token",
                "detail": "Add HUGGINGFACE_TOKEN to .env (huggingface.co → Settings → Tokens)", "usage_pct": None}
    try:
        t0 = time.time()
        resp = requests.get(
            "https://huggingface.co/api/whoami-v2",
            headers={"Authorization": f"Bearer {token}"},
            timeout=8,
        )
        latency = int((time.time() - t0) * 1000)
        if resp.status_code == 200:
            data = resp.json()
            plan = data.get("auth", {}).get("accessToken", {}).get("role", "read")
            user = data.get("name", "unknown")
            is_pro = "pro" in str(data).lower() or "enterprise" in str(data).lower()
            note_extra = "✅ Pro plan — FLUX/SDXL available" if is_pro else "⚠️ Free plan — FLUX/SDXL need Pro, using Openverse fallback"
            return {**base, "status": "active" if is_pro else "degraded",
                    "status_text": "Pro Active" if is_pro else "Free (Limited)",
                    "latency_ms": latency, "usage_pct": None,
                    "detail": f"User: {user} · {note_extra}"}
        if resp.status_code == 401:
            return {**base, "status": "error", "status_text": "Invalid Token",
                    "detail": "Token rejected — regenerate at huggingface.co/settings/tokens", "usage_pct": None}
        return {**base, "status": "error", "status_text": f"HTTP {resp.status_code}",
                "detail": resp.text[:120], "usage_pct": None}
    except Exception as e:
        return {**base, "status": "error", "status_text": "Error", "detail": str(e)[:120], "usage_pct": None}


def check_pollinations() -> dict:
    base = {
        "id": "pollinations", "name": "Pollinations.ai", "type": "Image Generation (Free)",
        "icon": "🌸", "key_preview": "No key needed",
        "note": "Free · 1 concurrent request per IP · Falls back to Openverse on 402",
        "checked_at": _now(),
    }
    try:
        t0 = time.time()
        resp = requests.get("https://image.pollinations.ai/models", timeout=8)
        latency = int((time.time() - t0) * 1000)
        if resp.status_code == 200:
            models = resp.json() if isinstance(resp.json(), list) else []
            return {**base, "status": "active", "status_text": "Online",
                    "latency_ms": latency, "usage_pct": None,
                    "detail": f"Service online · {len(models)} models · Latency: {latency}ms"}
        return {**base, "status": "rate_limited", "status_text": "Queue Full",
                "usage_pct": 100, "usage_text": "IP queue saturated",
                "detail": "Shared IP rate limit — Openverse fallback active"}
    except Exception as e:
        return {**base, "status": "error", "status_text": "Unreachable",
                "detail": f"Cannot reach pollinations.ai: {str(e)[:80]}", "usage_pct": None}


def check_openverse() -> dict:
    base = {
        "id": "openverse", "name": "Openverse", "type": "Image Search (Free)",
        "icon": "🖼️", "key_preview": "No key needed",
        "note": "Free Creative Commons photos from Flickr, Wikipedia & more",
        "checked_at": _now(),
    }
    try:
        t0 = time.time()
        resp = requests.get("https://api.openverse.org/v1/images/?q=news&page_size=1", timeout=8)
        latency = int((time.time() - t0) * 1000)
        if resp.status_code == 200:
            count = resp.json().get("count", 0)
            return {**base, "status": "active", "status_text": "Active",
                    "latency_ms": latency, "usage_pct": None,
                    "detail": f"Connected · {count:,}+ images indexed · Latency: {latency}ms"}
        return {**base, "status": "error", "status_text": f"HTTP {resp.status_code}",
                "detail": resp.text[:100], "usage_pct": None}
    except Exception as e:
        return {**base, "status": "error", "status_text": "Unreachable",
                "detail": str(e)[:120], "usage_pct": None}


def check_cloudinary() -> dict:
    cloud = os.getenv("CLOUDINARY_CLOUD_NAME", "")
    api_key = os.getenv("CLOUDINARY_API_KEY", "")
    api_secret = os.getenv("CLOUDINARY_API_SECRET", "")
    base = {
        "id": "cloudinary", "name": "Cloudinary", "type": "Video Hosting",
        "icon": "☁️", "key_preview": cloud or _mask(api_key),
        "note": "Free: 25 GB storage, 25 GB bandwidth/month · Video hosting for Instagram API",
        "checked_at": _now(),
    }
    if not cloud:
        return {**base, "status": "not_configured", "status_text": "Not Configured",
                "detail": "Add CLOUDINARY_CLOUD_NAME to .env", "usage_pct": None}
    try:
        t0 = time.time()
        resp = requests.get(f"https://res.cloudinary.com/{cloud}/image/upload/sample.jpg", timeout=8)
        latency = int((time.time() - t0) * 1000)
        if resp.status_code in (200, 404):
            return {**base, "status": "active", "status_text": "Active",
                    "latency_ms": latency, "usage_pct": None,
                    "detail": f"Cloud: {cloud} · Reachable · Latency: {latency}ms"}
        return {**base, "status": "error", "status_text": f"HTTP {resp.status_code}",
                "detail": f"Cloud name: {cloud}", "usage_pct": None}
    except Exception as e:
        return {**base, "status": "error", "status_text": "Error",
                "detail": str(e)[:120], "usage_pct": None}


def check_facebook() -> dict:
    token   = os.getenv("FACEBOOK_ACCESS_TOKEN", "")
    page_id = os.getenv("FACEBOOK_PAGE_ID", "")
    base = {
        "id": "facebook", "name": "Facebook Graph API", "type": "Publishing",
        "icon": "📘", "key_preview": _mask(token),
        "note": f"Page ID: {page_id or 'not set'} · Posts videos to Facebook Page",
        "checked_at": _now(),
    }
    if not token or not page_id:
        missing = []
        if not token:   missing.append("FACEBOOK_ACCESS_TOKEN")
        if not page_id: missing.append("FACEBOOK_PAGE_ID")
        return {**base, "status": "not_configured", "status_text": "Not Configured",
                "detail": f"Missing: {', '.join(missing)}", "usage_pct": None}
    try:
        t0 = time.time()
        resp = requests.get(
            f"https://graph.facebook.com/{page_id}",
            params={"fields": "id,name", "access_token": token},
            timeout=8,
        )
        latency = int((time.time() - t0) * 1000)
        if resp.status_code == 200:
            data = resp.json()
            name = data.get("name", "unknown")
            return {**base, "status": "active", "status_text": "Active",
                    "latency_ms": latency, "usage_pct": None,
                    "detail": f"Page: {name} (ID: {page_id}) · Token valid · Latency: {latency}ms"}
        data = resp.json()
        err  = data.get("error", {})
        if err.get("code") == 190:
            return {**base, "status": "error", "status_text": "Token Expired",
                    "detail": "Refresh your Facebook access token in Meta Developer Console", "usage_pct": None}
        return {**base, "status": "error", "status_text": f"HTTP {resp.status_code}",
                "detail": err.get("message", resp.text[:100]), "usage_pct": None}
    except Exception as e:
        return {**base, "status": "error", "status_text": "Error", "detail": str(e)[:120], "usage_pct": None}


def check_instagram() -> dict:
    token      = os.getenv("FACEBOOK_ACCESS_TOKEN", "")
    account_id = os.getenv("INSTAGRAM_BUSINESS_ACCOUNT_ID", "")
    base = {
        "id": "instagram", "name": "Instagram Graph API", "type": "Publishing",
        "icon": "📸", "key_preview": account_id or _mask(token),
        "note": f"Account ID: {account_id or 'not set'} · Posts Reels to Instagram Business",
        "checked_at": _now(),
    }
    if not token or not account_id:
        missing = []
        if not token:      missing.append("FACEBOOK_ACCESS_TOKEN")
        if not account_id: missing.append("INSTAGRAM_BUSINESS_ACCOUNT_ID")
        return {**base, "status": "not_configured", "status_text": "Not Configured",
                "detail": f"Missing: {', '.join(missing)}", "usage_pct": None}
    try:
        t0 = time.time()
        resp = requests.get(
            f"https://graph.facebook.com/{account_id}",
            params={"fields": "id,name,username", "access_token": token},
            timeout=8,
        )
        latency = int((time.time() - t0) * 1000)
        if resp.status_code == 200:
            data = resp.json()
            username = data.get("username", data.get("name", "unknown"))
            return {**base, "status": "active", "status_text": "Active",
                    "latency_ms": latency, "usage_pct": None,
                    "detail": f"@{username} · Token valid · Latency: {latency}ms"}
        err = resp.json().get("error", {})
        if err.get("code") == 190:
            return {**base, "status": "error", "status_text": "Token Expired",
                    "detail": "Refresh FACEBOOK_ACCESS_TOKEN", "usage_pct": None}
        return {**base, "status": "error", "status_text": f"HTTP {resp.status_code}",
                "detail": err.get("message", "")[:120], "usage_pct": None}
    except Exception as e:
        return {**base, "status": "error", "status_text": "Error", "detail": str(e)[:120], "usage_pct": None}


def get_all_statuses() -> list[dict]:
    import concurrent.futures
    checkers = [
        check_groq, check_newsapi, check_huggingface,
        check_pollinations, check_openverse, check_cloudinary,
        check_facebook, check_instagram,
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(lambda f: f(), checkers))
    return results
