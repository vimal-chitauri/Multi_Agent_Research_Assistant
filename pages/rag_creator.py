import streamlit as st
import os
import json
import re
import requests
import html as html_module
from collections import Counter
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

try:
    import chromadb
    CHROMA_OK = True
except ImportError:
    CHROMA_OK = False

from llm.adapter import LLMAdapter

RAG_DB_DIR      = Path(__file__).parent.parent / "rag_db"
RAG_DB_DIR.mkdir(exist_ok=True)
COLLECTION_NAME = "social_posts"

TOKEN      = os.getenv("FACEBOOK_ACCESS_TOKEN", "")
PAGE_ID    = os.getenv("FACEBOOK_PAGE_ID", "")
ACCOUNT_ID = os.getenv("INSTAGRAM_BUSINESS_ACCOUNT_ID", "")
NICHE      = os.getenv("NICHE", "latest_news")
has_fb     = bool(TOKEN and PAGE_ID)
has_ig     = bool(TOKEN and ACCOUNT_ID)

st.set_page_config(
    page_title="RAG Creator — NewsFlow AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [data-testid="stApp"] {
    background:#0d0d1a !important; color:#e2e8f0 !important;
    font-family:'Inter',-apple-system,sans-serif !important;
}
header[data-testid="stHeader"]               { display:none !important; }
#MainMenu                                     { display:none !important; }
[data-testid="stSidebar"]                    { display:none !important; }
[data-testid="collapsedControl"]             { display:none !important; }
[data-testid="stSidebarNavItems"]            { display:none !important; }
section[data-testid="stSidebarNavSeparator"] { display:none !important; }
iframe[height="0"] {
    display:block !important; height:0 !important;
    overflow:hidden !important; margin:0 !important;
    padding:0 !important; border:none !important;
}
.main .block-container {
    padding-top:68px !important; padding-bottom:44px !important;
    padding-left:32px !important; padding-right:32px !important;
    max-width:100% !important;
}

/* ── Navbar ───────────────────────────────────────────────────── */
.nf-navbar {
    position:fixed; top:0; left:0; right:0; z-index:999999;
    height:52px; background:#07070f;
    border-bottom:1px solid rgba(139,92,246,0.16);
    display:flex; align-items:center; padding:0 28px; gap:0;
}
.nb-logo { display:flex; align-items:center; gap:9px; margin-right:28px; flex-shrink:0; }
.nb-logo-icon { font-size:20px; filter:drop-shadow(0 0 7px rgba(139,92,246,0.9)); }
.nb-logo-text { font-size:15px; font-weight:700; color:#e2e8f0; letter-spacing:-0.3px; white-space:nowrap; }
.nb-div { width:1px; height:22px; background:rgba(139,92,246,0.18); margin-right:6px; flex-shrink:0; }
.nb-item {
    display:inline-flex; align-items:center; padding:0 16px; height:52px;
    font-size:13px; font-weight:500; color:#6b7280;
    text-decoration:none !important; border-bottom:2px solid transparent;
    transition:color 0.15s, border-color 0.15s; white-space:nowrap; cursor:pointer;
}
.nb-item:hover { color:#c4b5fd; }
.nb-active { color:#e2e8f0 !important; border-bottom-color:#8b5cf6 !important; font-weight:600 !important; }

/* ── Buttons ─────────────────────────────────────────────────── */
[data-testid="stButton"] button[kind="primary"] {
    background:linear-gradient(135deg,#7c3aed,#6d28d9) !important;
    border:none !important; color:#fff !important; font-weight:600 !important;
    border-radius:8px !important;
}
[data-testid="stButton"] button[kind="secondary"] {
    background:rgba(139,92,246,0.08) !important;
    border:1px solid rgba(139,92,246,0.3) !important;
    color:#a78bfa !important; border-radius:8px !important; font-weight:500 !important;
}
[data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea {
    background:#0f0f1e !important; border:1px solid rgba(139,92,246,0.25) !important;
    border-radius:8px !important; color:#e2e8f0 !important;
}
[data-testid="stSelectbox"] > div > div {
    background:#0f0f1e !important; border:1px solid rgba(139,92,246,0.25) !important;
    border-radius:8px !important; color:#e2e8f0 !important;
}

/* ── Cards ───────────────────────────────────────────────────── */
.rag-card {
    background:#13131f; border:1px solid rgba(139,92,246,0.15); border-radius:12px;
    padding:18px 20px; margin-bottom:14px;
}
.rag-card-title {
    font-size:12px; font-weight:700; color:#6d28d9; letter-spacing:0.6px;
    text-transform:uppercase; margin-bottom:12px;
}
.rag-section-title {
    font-size:11px; font-weight:600; color:#4b5563; letter-spacing:0.5px;
    text-transform:uppercase; margin-bottom:6px;
}

/* ── Style profile tags ──────────────────────────────────────── */
.tag-chip {
    display:inline-block; padding:3px 10px; border-radius:20px;
    font-size:11px; font-weight:600; margin:2px 3px;
    background:rgba(139,92,246,0.12); color:#a78bfa;
    border:1px solid rgba(139,92,246,0.22);
}

/* ── Generated output ────────────────────────────────────────── */
.gen-heading {
    font-size:20px; font-weight:800; color:#e2e8f0; letter-spacing:-0.3px;
    line-height:1.3; margin-bottom:14px;
    background:linear-gradient(135deg,#a78bfa,#c084fc);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
.gen-caption-box {
    background:#0f0f1e; border:1px solid rgba(139,92,246,0.2); border-radius:10px;
    padding:16px; font-size:13px; color:#d1d5db; line-height:1.65;
    white-space:pre-wrap; word-wrap:break-word; margin-bottom:14px;
    max-height:280px; overflow-y:auto;
}
.gen-image-prompt {
    background:#080812; border:1px solid rgba(99,102,241,0.2); border-radius:8px;
    padding:12px 14px; font-family:'JetBrains Mono',monospace; font-size:11px;
    color:#818cf8; line-height:1.5; margin-bottom:14px; word-wrap:break-word;
}
.gen-meta-row {
    display:flex; gap:16px; flex-wrap:wrap; margin-bottom:14px;
}
.gen-meta-item {
    background:#13131f; border:1px solid rgba(139,92,246,0.12);
    border-radius:8px; padding:8px 14px;
    font-size:12px; color:#9ca3af;
}
.gen-meta-item strong { color:#c4b5fd; font-weight:600; }

/* ── Similar post mini-card ──────────────────────────────────── */
.sim-card {
    background:#0c0c1a; border:1px solid rgba(139,92,246,0.12); border-radius:8px;
    padding:10px 14px; margin-bottom:8px;
    display:flex; gap:12px; align-items:flex-start;
}
.sim-score {
    font-size:11px; font-weight:700; color:#7c3aed;
    background:rgba(124,58,237,0.12); padding:2px 8px; border-radius:12px;
    white-space:nowrap; flex-shrink:0;
}
.sim-text {
    font-size:11px; color:#6b7280; line-height:1.45;
    display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden;
}
.sim-meta { font-size:10px; color:#4b5563; margin-top:4px; }

/* ── Divider ─────────────────────────────────────────────────── */
.rag-divider { border:none; border-top:1px solid rgba(139,92,246,0.1); margin:16px 0; }

/* ── Metric chips ─────────────────────────────────────────────── */
.kb-metric {
    background:#13131f; border:1px solid rgba(139,92,246,0.15); border-radius:10px;
    padding:12px 16px; text-align:center;
}
.kb-metric-val { font-size:22px; font-weight:800; color:#a78bfa; }
.kb-metric-label { font-size:10px; color:#6b7280; margin-top:2px; text-transform:uppercase; letter-spacing:0.4px; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    pass

st.iframe("""<script>
(function() {
  var p = window.parent.document;
  var old = p.getElementById("nf-topnav");
  if (old) old.remove();
  var nav = p.createElement("div");
  nav.id = "nf-topnav";
  nav.className = "nf-navbar";
  nav.innerHTML = `
    <div class="nb-logo">
      <span class="nb-logo-icon">&#9889;</span>
      <span class="nb-logo-text">NewsFlow AI</span>
    </div>
    <div class="nb-div"></div>
    <span class="nb-item" onclick="window.location.href='/'">NewsFlow AI</span>
    <span class="nb-item" onclick="window.location.href='/api_status'">API Status Dashboard</span>
    <span class="nb-item" onclick="window.location.href='/history'">History</span>
    <span class="nb-item nb-active">RAG Creator</span>
  `;
  p.body.prepend(nav);
})();
</script>""", height=1)


# ChromaDB helpers

@st.cache_resource(show_spinner=False)
def _get_collection():
    from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
    ef     = ONNXMiniLM_L6_V2()   # downloads ~80 MB model on first call (cached to ~/.cache/chroma)
    client = chromadb.PersistentClient(path=str(RAG_DB_DIR))
    # If the collection already exists but was created without an embedding function
    # (e.g. first-run race), delete and recreate it so queries work properly.
    try:
        existing = client.get_collection(COLLECTION_NAME)
        # Try a dummy query to verify the embedding function is wired up
        if existing.count() > 0:
            existing.query(query_texts=["test"], n_results=1)
        col = existing
    except Exception:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        col = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )
    return col


# API fetch (mirrors history.py)

def _fetch_instagram(limit: int = 100) -> tuple[list, str | None]:
    try:
        r = requests.get(
            f"https://graph.facebook.com/{ACCOUNT_ID}/media",
            params={
                "fields": "id,caption,media_type,media_url,thumbnail_url,timestamp,permalink,like_count,comments_count",
                "limit": limit,
                "access_token": TOKEN,
            },
            timeout=15,
        )
        data = r.json()
        if r.status_code == 200:
            posts = data.get("data", [])
            for p in posts:
                p["_platform"] = "instagram"
            return posts, None
        return [], data.get("error", {}).get("message", r.text[:120])
    except Exception as e:
        return [], str(e)


def _fetch_facebook(limit: int = 100) -> tuple[list, str | None]:
    try:
        r = requests.get(
            f"https://graph.facebook.com/{PAGE_ID}/posts",
            params={
                "fields": "id,message,created_time,full_picture,permalink_url,"
                          "reactions.summary(true),comments.summary(true),shares",
                "limit": limit,
                "access_token": TOKEN,
            },
            timeout=15,
        )
        data = r.json()
        if r.status_code == 200:
            posts = data.get("data", [])
            for p in posts:
                p["_platform"] = "facebook"
            return posts, None
        return [], data.get("error", {}).get("message", r.text[:120])
    except Exception as e:
        return [], str(e)


# Vector DB operations

def _extract_hashtags(text: str) -> list[str]:
    return re.findall(r"#\w+", text or "")


def sync_posts_to_db(col, limit: int = 100) -> tuple[int, list[str]]:
    """Fetch posts from IG + FB, upsert into ChromaDB. Returns (indexed_count, errors)."""
    all_posts: list[dict] = []
    errors: list[str] = []

    if has_ig:
        posts, err = _fetch_instagram(limit)
        if err:
            errors.append(f"Instagram: {err}")
        else:
            all_posts += posts
    if has_fb:
        posts, err = _fetch_facebook(limit)
        if err:
            errors.append(f"Facebook: {err}")
        else:
            all_posts += posts

    indexed = 0
    for post in all_posts:
        platform = post.get("_platform", "")
        if platform == "instagram":
            pid      = post.get("id", "")
            caption  = (post.get("caption") or "").strip()
            likes    = int(post.get("like_count") or 0)
            comments = int(post.get("comments_count") or 0)
            img_url  = post.get("thumbnail_url") or post.get("media_url") or ""
            ts       = post.get("timestamp", "")
            link     = post.get("permalink", "")
        else:
            pid      = post.get("id", "")
            caption  = (post.get("message") or "").strip()
            likes    = int(post.get("reactions", {}).get("summary", {}).get("total_count") or 0)
            comments = int(post.get("comments", {}).get("summary", {}).get("total_count") or 0)
            img_url  = post.get("full_picture") or ""
            ts       = post.get("created_time", "")
            link     = post.get("permalink_url", "")

        if not caption:
            continue

        tags = _extract_hashtags(caption)
        doc_id = f"{platform}_{pid}"
        try:
            col.upsert(
                ids=[doc_id],
                documents=[caption],
                metadatas=[{
                    "platform":    platform,
                    "post_id":     pid,
                    "likes":       likes,
                    "comments":    comments,
                    "engagement":  likes + comments,
                    "has_image":   int(bool(img_url)),
                    "image_url":   img_url[:500],
                    "permalink":   link[:500],
                    "timestamp":   ts[:20],
                    "hashtags":    json.dumps(tags[:20]),
                    "caption_len": len(caption),
                }],
            )
            indexed += 1
        except Exception:
            pass

    return indexed, errors


def compute_style_profile(col) -> dict:
    """Derive style insights from all indexed posts in the collection."""
    try:
        res   = col.get(include=["documents", "metadatas"])
        docs  = res.get("documents") or []
        metas = res.get("metadatas") or []
    except Exception:
        return {}

    total = len(docs)
    if total == 0:
        return {}

    all_tags:    list[str] = []
    all_hours:   list[int] = []
    total_eng    = 0
    total_cap    = 0
    total_img    = 0
    platforms    = Counter()

    for m in metas:
        total_eng  += m.get("engagement", 0)
        total_cap  += m.get("caption_len", 0)
        total_img  += m.get("has_image", 0)
        platforms[m.get("platform", "?")] += 1

        tags = json.loads(m.get("hashtags", "[]"))
        all_tags.extend(tags)

        ts = m.get("timestamp", "")
        if ts and len(ts) >= 13:
            try:
                h = int(ts[11:13])
                all_hours.append(h)
            except Exception:
                pass

    tag_counts  = Counter(all_tags).most_common(12)
    peak_hour   = Counter(all_hours).most_common(1)[0][0] if all_hours else None

    # Top 3 posts by engagement
    paired = sorted(zip(docs, metas), key=lambda x: x[1].get("engagement", 0), reverse=True)
    top3   = [{"caption": d, **m} for d, m in paired[:3]]

    return {
        "total_posts":     total,
        "avg_engagement":  total_eng  // max(total, 1),
        "avg_caption_len": total_cap  // max(total, 1),
        "image_rate":      int(100 * total_img / max(total, 1)),
        "top_hashtags":    [t for t, _ in tag_counts],
        "platforms":       dict(platforms),
        "peak_hour":       peak_hour,
        "top3_posts":      top3,
    }


def find_similar_posts(col, query: str, n: int = 5) -> list[dict]:
    """Cosine-similarity search for posts closest to query text."""
    count = col.count()
    if count == 0:
        return []
    try:
        res = col.query(
            query_texts=[query],
            n_results=min(n, count),
            include=["documents", "metadatas", "distances"],
        )
        similar = []
        for doc, meta, dist in zip(
            res["documents"][0],
            res["metadatas"][0],
            res["distances"][0],
        ):
            similar.append({
                "caption":    doc,
                "similarity": round(max(0.0, 1.0 - dist), 3),
                **meta,
            })
        return similar
    except Exception:
        return []


# RAG-powered content generation

def generate_rag_post(topic: str, similar: list[dict], profile: dict) -> dict:
    """Call Groq LLM with RAG context to create a new post in the account's style."""
    llm = LLMAdapter(model="smart")

    examples = "\n\n".join(
        f"EXAMPLE {i+1}  (❤️ {p.get('likes',0):,}  💬 {p.get('comments',0):,}):\n{p['caption'][:500]}"
        for i, p in enumerate(similar[:4])
    ) or "No past posts found — generate a fresh, engaging post."

    avg_len  = profile.get("avg_caption_len", 250)
    top_tags = profile.get("top_hashtags", [])[:8]
    peak_h   = profile.get("peak_hour")
    peak_str = (f"{peak_h}:00 – {peak_h+1}:00 UTC" if peak_h is not None else "unknown")

    system = (
        "You are an expert social media copywriter who has deeply studied this account. "
        "You write new content that feels 100% authentic — same voice, same structure, same energy. "
        "Always respond with valid JSON only. No extra text."
    )
    prompt = f"""
TOPIC FOR NEW POST: {topic}

STYLE REFERENCE — Study these past posts from this account and learn the tone, structure, voice:

{examples}

ACCOUNT STATISTICS:
- Total indexed posts : {profile.get("total_posts", 0)}
- Avg caption length  : {avg_len} characters
- Avg engagement      : {profile.get("avg_engagement", 0):,} (likes + comments)
- Most-used hashtags  : {", ".join(top_tags) if top_tags else "none detected"}
- Peak posting hour   : {peak_str}
- Image rate          : {profile.get("image_rate", 0)}% of posts have images

TASK: Write a brand-new post about "{topic}" that:
1. Perfectly matches the voice and tone of the examples above
2. Uses a similar caption structure (strong hook → value → CTA)
3. Mirrors the hashtag style and count this account uses
4. Feels like it was written by the same person, not an AI

Respond ONLY with this JSON object:
{{
  "heading": "Bold punchy headline under 10 words — ALL CAPS style",
  "caption": "Full caption — approx {avg_len} chars — written in this account's exact voice. Use \\n for line breaks.",
  "hashtags": ["list", "of", "hashtags", "matching", "this", "account's", "style", "10-12 total"],
  "image_prompt": "Detailed AI image prompt for this topic. No real people/faces/names. Cinematic quality, dramatic lighting, professional photography style.",
  "music_suggestion": "Short description of music energy/genre that fits this post's mood",
  "posting_tip": "Best day and time recommendation based on this account's peak hour data"
}}
"""
    raw = llm.chat(system, prompt, temperature=0.75)
    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        return json.loads(cleaned.strip())
    except Exception:
        return {"error": "Could not parse generated content", "raw": raw[:600]}


# UI helpers

def _tag_chips(tags: list[str]) -> str:
    return "".join(f'<span class="tag-chip">#{t.lstrip("#")}</span>' for t in tags)


def _fmt_hour(h: int | None) -> str:
    if h is None:
        return "—"
    suffix = "AM" if h < 12 else "PM"
    disp   = h if h <= 12 else h - 12
    disp   = 12 if disp == 0 else disp
    return f"{disp}:00 {suffix}"


def _platform_badge(platform: str) -> str:
    if platform == "instagram":
        return '<span style="color:#e879f9;font-size:11px;">📸 Instagram</span>'
    return '<span style="color:#60a5fa;font-size:11px;">📘 Facebook</span>'


# PAGE LAYOUT

st.markdown("""
<div style="margin-bottom:6px;">
  <div style="font-size:22px;font-weight:700;color:#e2e8f0;margin-bottom:3px;">
    🧠 RAG Creator
  </div>
  <div style="font-size:13px;color:#6b7280;">
    Learns from your past 100 posts · generates new content in your exact voice and style
  </div>
</div>
<hr class="rag-divider">
""", unsafe_allow_html=True)

if not CHROMA_OK:
    st.error("ChromaDB is not installed. Run: `pip install chromadb onnxruntime` then restart.")
    st.stop()

if not has_fb and not has_ig:
    st.warning("No social accounts configured. Add FACEBOOK_ACCESS_TOKEN and account IDs to your .env file.")
    st.stop()

with st.spinner("🔌 Connecting to knowledge base… (downloading AI model on first visit — ~80 MB, one-time)"):
    _col = _get_collection()
_kb_count = _col.count()

if _kb_count == 0 and not st.session_state.get("rag_auto_synced"):
    st.session_state["rag_auto_synced"] = True
    with st.spinner("📥 Auto-syncing your last 100 posts into the knowledge base…"):
        _n_auto, _auto_errs = sync_posts_to_db(_col, limit=100)
    for _e in _auto_errs:
        st.warning(_e)
    if _n_auto > 0:
        _kb_count = _col.count()
        st.success(f"✅ {_n_auto} posts indexed — knowledge base is ready!")
        st.rerun()
    else:
        st.info("No posts with captions found yet. Run your pipeline and publish some posts first, then come back.")

_profile = compute_style_profile(_col) if _kb_count > 0 else {}

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.markdown(
        f'<div class="kb-metric"><div class="kb-metric-val">{_kb_count}</div>'
        f'<div class="kb-metric-label">Posts Indexed</div></div>',
        unsafe_allow_html=True)
with k2:
    st.markdown(
        f'<div class="kb-metric"><div class="kb-metric-val">{_profile.get("avg_engagement", 0):,}</div>'
        f'<div class="kb-metric-label">Avg Engagement</div></div>',
        unsafe_allow_html=True)
with k3:
    st.markdown(
        f'<div class="kb-metric"><div class="kb-metric-val">{_profile.get("avg_caption_len", 0)}</div>'
        f'<div class="kb-metric-label">Avg Caption Chars</div></div>',
        unsafe_allow_html=True)
with k4:
    st.markdown(
        f'<div class="kb-metric"><div class="kb-metric-val">{_profile.get("image_rate", 0)}%</div>'
        f'<div class="kb-metric-label">Posts With Image</div></div>',
        unsafe_allow_html=True)
with k5:
    st.markdown(
        f'<div class="kb-metric"><div class="kb-metric-val">{_fmt_hour(_profile.get("peak_hour"))}</div>'
        f'<div class="kb-metric-label">Peak Post Hour</div></div>',
        unsafe_allow_html=True)

st.markdown('<hr class="rag-divider">', unsafe_allow_html=True)

# Two-column main layout
_left, _right = st.columns([3, 2], gap="large")

# LEFT — Create new post
with _left:
    st.markdown('<div class="rag-card-title">✨ Create New Post</div>', unsafe_allow_html=True)

    _topic = st.text_input(
        "Topic or prompt",
        placeholder="e.g. FIFA World Cup 2026 semifinal results",
        key="rag_topic_input",
    )

    _c1, _c2 = st.columns(2)
    with _c1:
        _n_similar = st.selectbox(
            "Style reference",
            options=[3, 5, 10],
            format_func=lambda x: f"Use top {x} similar past posts",
            key="rag_n_similar",
        )
    with _c2:
        _niche_override = st.selectbox(
            "Niche",
            options=["latest_news", "technology", "finance", "fitness",
                     "motivation", "crypto", "marketing", "food", "travel"],
            index=["latest_news", "technology", "finance", "fitness",
                   "motivation", "crypto", "marketing", "food", "travel"].index(NICHE)
                  if NICHE in ["latest_news", "technology", "finance", "fitness",
                               "motivation", "crypto", "marketing", "food", "travel"] else 0,
            key="rag_niche",
        )

    _gen_btn = st.button(
        "🚀 Generate RAG-Powered Post",
        type="primary",
        width='stretch',
        disabled=(_kb_count == 0 or not _topic.strip()),
        key="rag_generate_btn",
    )

    if _gen_btn and _topic.strip():
        with st.spinner("🔍 Searching your knowledge base for similar posts…"):
            _similar = find_similar_posts(_col, _topic.strip(), n=_n_similar)

        with st.spinner("🧠 Generating content in your style…"):
            _result = generate_rag_post(_topic.strip(), _similar, _profile)

        st.session_state["rag_result"]  = _result
        st.session_state["rag_similar"] = _similar
        st.session_state["rag_used_topic"] = _topic.strip()
        st.rerun()

    _result  = st.session_state.get("rag_result")
    _similar = st.session_state.get("rag_similar", [])

    if _result:
        st.markdown('<hr class="rag-divider">', unsafe_allow_html=True)

        if "error" in _result:
            st.error(f"Generation failed: {_result.get('error')}")
            if _result.get("raw"):
                with st.expander("Raw LLM output"):
                    st.text(_result["raw"])
        else:
            _used_topic = st.session_state.get("rag_used_topic", "")

            # Heading
            st.markdown(
                f'<div class="gen-heading">{html_module.escape(_result.get("heading", ""))}</div>',
                unsafe_allow_html=True)

            # Caption
            _cap = _result.get("caption", "")
            st.markdown('<div class="rag-section-title">Caption</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="gen-caption-box">{html_module.escape(_cap)}</div>',
                unsafe_allow_html=True)

            # Copy caption button
            _cc1, _cc2 = st.columns([3, 1])
            with _cc2:
                if st.button("📋 Copy", key="copy_caption", width='stretch'):
                    st.write("Caption ready — use Ctrl+C after selecting above")

            # Hashtags
            _tags = _result.get("hashtags", [])
            if _tags:
                st.markdown('<div class="rag-section-title" style="margin-top:10px;">Hashtags</div>',
                            unsafe_allow_html=True)
                st.markdown(_tag_chips(_tags), unsafe_allow_html=True)

            # Meta row
            st.markdown('<div style="margin-top:14px;"></div>', unsafe_allow_html=True)
            _music = _result.get("music_suggestion", "")
            _tip   = _result.get("posting_tip", "")
            _music_html = (f'<div class="gen-meta-item"><strong>🎵 Music</strong><br>{html_module.escape(_music)}</div>'
                           if _music else "")
            _tip_html   = (f'<div class="gen-meta-item"><strong>📅 Post time</strong><br>{html_module.escape(_tip)}</div>'
                           if _tip else "")
            st.markdown(
                f'<div class="gen-meta-row">{_music_html}{_tip_html}</div>',
                unsafe_allow_html=True)

            # Image prompt
            _img_p = _result.get("image_prompt", "")
            if _img_p:
                st.markdown('<div class="rag-section-title">AI Image Prompt</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="gen-image-prompt">{html_module.escape(_img_p)}</div>',
                    unsafe_allow_html=True)

            # Similar posts used
            if _similar:
                with st.expander(f"📎 {len(_similar)} similar past posts used as style reference"):
                    for sp in _similar:
                        _sim_pct = int(sp.get("similarity", 0) * 100)
                        _plat    = sp.get("platform", "")
                        _eng     = sp.get("engagement", 0)
                        _cap_s   = sp.get("caption", "")[:180]
                        st.markdown(
                            f'<div class="sim-card">'
                            f'<div>'
                            f'<span class="sim-score">{_sim_pct}% match</span>'
                            f'</div>'
                            f'<div style="flex:1;">'
                            f'<div class="sim-text">{html_module.escape(_cap_s)}{"…" if len(sp.get("caption",""))>180 else ""}</div>'
                            f'<div class="sim-meta">'
                            f'{_platform_badge(_plat)} &nbsp; ❤️ {sp.get("likes",0):,} &nbsp; 💬 {sp.get("comments",0):,}</div>'
                            f'</div>'
                            f'</div>',
                            unsafe_allow_html=True)

            # Action hint
            st.markdown(
                '<div style="font-size:11px;color:#4b5563;margin-top:8px;">'
                '💡 Use this image prompt in the main pipeline → Images tab for AI-generated visuals'
                '</div>',
                unsafe_allow_html=True)


# RIGHT — Knowledge Base + Style Profile
with _right:

    st.markdown('<div class="rag-card-title">🗄️ Knowledge Base</div>', unsafe_allow_html=True)

    _r1, _r2 = st.columns([2, 1])
    with _r1:
        _sync_limit = st.select_slider(
            "Posts to index",
            options=[25, 50, 100, 200],
            value=100,
            format_func=lambda x: f"Last {x} posts",
            label_visibility="collapsed",
            key="rag_sync_limit",
        )
    with _r2:
        _sync_btn = st.button("🔄 Sync Posts", width='stretch', key="rag_sync_btn")

    if _sync_btn:
        with st.spinner(f"Fetching and indexing last {_sync_limit} posts…"):
            _n_indexed, _sync_errs = sync_posts_to_db(_col, limit=_sync_limit)
        if _sync_errs:
            for e in _sync_errs:
                st.warning(e)
        st.success(f"✅ {_n_indexed} posts indexed into knowledge base")
        st.rerun()

    if _kb_count == 0:
        st.markdown(
            '<div style="background:#0c0c1a;border:1px dashed rgba(139,92,246,0.2);'
            'border-radius:10px;padding:20px;text-align:center;color:#4b5563;font-size:12px;">'
            '<div style="font-size:28px;margin-bottom:8px;">📭</div>'
            'Click <strong style="color:#a78bfa">Sync Posts</strong> to index your past content<br>'
            'and enable RAG-powered generation'
            '</div>',
            unsafe_allow_html=True)
    else:
        # Platform split
        _plats = _profile.get("platforms", {})
        _ig_c  = _plats.get("instagram", 0)
        _fb_c  = _plats.get("facebook", 0)
        st.markdown(
            f'<div style="font-size:11px;color:#6b7280;margin:8px 0 16px 0;">'
            f'📸 {_ig_c} Instagram &nbsp;·&nbsp; 📘 {_fb_c} Facebook &nbsp;·&nbsp; '
            f'last synced just now</div>',
            unsafe_allow_html=True)

        st.markdown('<hr class="rag-divider">', unsafe_allow_html=True)
        st.markdown('<div class="rag-card-title">🎨 Your Style Profile</div>', unsafe_allow_html=True)

        # Top hashtags
        _top_tags = _profile.get("top_hashtags", [])
        if _top_tags:
            st.markdown('<div class="rag-section-title">Top Hashtags</div>', unsafe_allow_html=True)
            st.markdown(_tag_chips(_top_tags[:10]), unsafe_allow_html=True)

        # Caption length
        _avg_c = _profile.get("avg_caption_len", 0)
        if _avg_c:
            st.markdown(
                f'<div style="margin-top:14px;" class="rag-section-title">Caption Length</div>'
                f'<div style="font-size:13px;color:#d1d5db;">'
                f'Your captions average <strong style="color:#a78bfa">{_avg_c} characters</strong> '
                f'({"short & punchy" if _avg_c < 200 else "medium detail" if _avg_c < 400 else "long-form storytelling"})</div>',
                unsafe_allow_html=True)

        # Peak hour
        _ph = _profile.get("peak_hour")
        if _ph is not None:
            st.markdown(
                f'<div style="margin-top:12px;" class="rag-section-title">Best Post Time</div>'
                f'<div style="font-size:13px;color:#d1d5db;">'
                f'Your posts perform best around <strong style="color:#a78bfa">{_fmt_hour(_ph)}</strong></div>',
                unsafe_allow_html=True)

        # Engagement
        _avg_eng = _profile.get("avg_engagement", 0)
        if _avg_eng:
            st.markdown(
                f'<div style="margin-top:12px;" class="rag-section-title">Avg Engagement</div>'
                f'<div style="font-size:13px;color:#d1d5db;">'
                f'<strong style="color:#a78bfa">{_avg_eng:,}</strong> likes + comments per post</div>',
                unsafe_allow_html=True)

        # Image rate
        _ir = _profile.get("image_rate", 0)
        st.markdown(
            f'<div style="margin-top:12px;" class="rag-section-title">Visual Content</div>'
            f'<div style="font-size:13px;color:#d1d5db;">'
            f'<strong style="color:#a78bfa">{_ir}%</strong> of your posts include images</div>',
            unsafe_allow_html=True)

        # Top 3 posts
        _top3 = _profile.get("top3_posts", [])
        if _top3:
            st.markdown('<hr class="rag-divider">', unsafe_allow_html=True)
            st.markdown('<div class="rag-card-title">🏆 Your Top 3 Posts</div>', unsafe_allow_html=True)
            for _i, _tp in enumerate(_top3, 1):
                _tc      = (_tp.get("caption") or "")[:120]
                _tp_plat = _tp.get("platform", "")
                _tl      = _tp.get("permalink", "#")
                _medal   = "🥇" if _i == 1 else "🥈" if _i == 2 else "🥉"
                _ellip   = "…" if len(_tp.get("caption", "")) > 120 else ""
                _vlink   = (
                    f'&nbsp; <a href="{html_module.escape(_tl)}" target="_blank"'
                    f' style="color:#8b5cf6;font-size:10px;">View ↗</a>'
                    if _tl and _tl != "#" else ""
                )
                st.markdown(
                    f'<div class="sim-card">'
                    f'<div style="font-size:16px;margin-top:2px;">{_medal}</div>'
                    f'<div style="flex:1;">'
                    f'<div class="sim-text">{html_module.escape(_tc)}{_ellip}</div>'
                    f'<div class="sim-meta">'
                    f'{_platform_badge(_tp_plat)} &nbsp; '
                    f'❤️ {_tp.get("likes",0):,} &nbsp; 💬 {_tp.get("comments",0):,}'
                    f'{_vlink}'
                    f'</div></div></div>',
                    unsafe_allow_html=True)
