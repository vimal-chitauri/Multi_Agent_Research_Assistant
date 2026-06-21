import streamlit as st
import os
import requests
import html as html_module
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(override=True)

st.set_page_config(
    page_title="History — NewsFlow AI",
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
iframe[height="0"] { display:block !important; height:0 !important; overflow:hidden !important; margin:0 !important; padding:0 !important; border:none !important; }
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

/* ── Page controls ────────────────────────────────────────────── */
[data-testid="stButton"] button[kind="secondary"] {
    background:rgba(139,92,246,0.08) !important; border:1px solid rgba(139,92,246,0.3) !important;
    color:#a78bfa !important; border-radius:8px !important; font-weight:500 !important;
}
[data-testid="stSelectbox"] > div > div {
    background:#0f0f1e !important; border:1px solid rgba(139,92,246,0.25) !important;
    border-radius:8px !important; color:#e2e8f0 !important;
}

/* ── Post cards ──────────────────────────────────────────────── */
.hist-card {
    background:#13131f; border:1px solid rgba(139,92,246,0.15); border-radius:10px;
    overflow:hidden; transition:border-color 0.2s, box-shadow 0.2s; margin-bottom:2px;
}
.hist-card:hover {
    border-color:rgba(139,92,246,0.5); box-shadow:0 4px 18px rgba(139,92,246,0.14);
}
.hist-thumb {
    position:relative; width:100%; padding-top:72%; background:#0a0a14; overflow:hidden;
}
.hist-thumb-inner {
    position:absolute; top:0; left:0; width:100%; height:100%;
    display:flex; align-items:center; justify-content:center;
}
.hist-thumb-inner img { width:100%; height:100%; object-fit:cover; display:block; }
.hist-badge {
    position:absolute; top:6px; left:6px; font-size:9px; font-weight:700;
    padding:2px 7px; border-radius:20px; white-space:nowrap;
}
.hist-body { padding:8px 10px 10px; }
.hist-date { font-size:9px; color:#6b7280; margin-bottom:4px; }
.hist-cap  {
    font-size:11px; color:#d1d5db; line-height:1.45; margin-bottom:6px;
    display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden;
    min-height:30px;
}
.hist-stats { display:flex; gap:10px; font-size:10px; color:#9ca3af; margin-bottom:8px; flex-wrap:wrap; }
.hist-view-link {
    display:inline-flex; align-items:center; gap:4px; font-size:10px; font-weight:600;
    color:#8b5cf6; text-decoration:none; padding:4px 10px;
    border:1px solid rgba(139,92,246,0.3); border-radius:5px;
    background:rgba(139,92,246,0.06); transition:background 0.15s;
}
.hist-view-link:hover { background:rgba(139,92,246,0.18); }
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
    <span class="nb-item nb-active">History</span>
    <span class="nb-item" onclick="window.location.href='/rag_creator'" style="cursor:pointer;">RAG Creator</span>
  `;
  p.body.prepend(nav);
})();
</script>""", height=1)


TOKEN      = os.getenv("FACEBOOK_ACCESS_TOKEN", "")
PAGE_ID    = os.getenv("FACEBOOK_PAGE_ID", "")
ACCOUNT_ID = os.getenv("INSTAGRAM_BUSINESS_ACCOUNT_ID", "")
has_fb     = bool(TOKEN and PAGE_ID)
has_ig     = bool(TOKEN and ACCOUNT_ID)


def _fmt_date(raw: str) -> str:
    try:
        dt = datetime.fromisoformat(raw.replace("+0000", "+00:00"))
        return dt.strftime("%b %d, %Y  ·  %I:%M %p")
    except Exception:
        return raw[:10] if raw else "—"


def fetch_instagram(limit: int = 24) -> tuple[list, str | None]:
    try:
        r = requests.get(
            f"https://graph.facebook.com/{ACCOUNT_ID}/media",
            params={
                "fields": "id,caption,media_type,media_url,thumbnail_url,timestamp,permalink,like_count,comments_count",
                "limit": limit,
                "access_token": TOKEN,
            },
            timeout=12,
        )
        data = r.json()
        if r.status_code == 200:
            posts = data.get("data", [])
            for p in posts:
                p["_platform"] = "instagram"
                p["_sort_time"] = p.get("timestamp", "")
            return posts, None
        err = data.get("error", {}).get("message", r.text[:120])
        return [], err
    except Exception as e:
        return [], str(e)


def fetch_facebook(limit: int = 24) -> tuple[list, str | None]:
    try:
        r = requests.get(
            f"https://graph.facebook.com/{PAGE_ID}/posts",
            params={
                "fields": "id,message,created_time,full_picture,permalink_url,"
                          "reactions.summary(true),comments.summary(true),shares",
                "limit": limit,
                "access_token": TOKEN,
            },
            timeout=12,
        )
        data = r.json()
        if r.status_code == 200:
            posts = data.get("data", [])
            for p in posts:
                p["_platform"] = "facebook"
                p["_sort_time"] = p.get("created_time", "")
            return posts, None
        err = data.get("error", {}).get("message", r.text[:120])
        return [], err
    except Exception as e:
        return [], str(e)


def _card_html(post: dict) -> str:
    platform = post.get("_platform", "")

    if platform == "instagram":
        thumb    = post.get("thumbnail_url") or post.get("media_url") or ""
        caption  = post.get("caption") or ""
        date_str = _fmt_date(post.get("timestamp", ""))
        link     = post.get("permalink", "#")
        likes    = post.get("like_count", 0)
        comments = post.get("comments_count", 0)
        badge    = "📸 Instagram"
        b_color  = "#e879f9"
        b_bg     = "rgba(217,70,239,0.15)"
        b_border = "rgba(217,70,239,0.3)"
        extra    = ""
    else:
        thumb    = post.get("full_picture", "")
        caption  = post.get("message") or ""
        date_str = _fmt_date(post.get("created_time", ""))
        link     = post.get("permalink_url", "#")
        likes    = post.get("reactions", {}).get("summary", {}).get("total_count", 0)
        comments = post.get("comments", {}).get("summary", {}).get("total_count", 0)
        shares   = post.get("shares", {}).get("count", 0)
        badge    = "📘 Facebook"
        b_color  = "#60a5fa"
        b_bg     = "rgba(59,130,246,0.15)"
        b_border = "rgba(59,130,246,0.3)"
        extra    = f'<span>🔁 {shares:,}</span>' if shares else ""

    cap_short = html_module.escape(caption[:90]) + ("…" if len(caption) > 90 else "")
    thumb_img = (
        f'<img src="{html_module.escape(thumb)}" alt="" style="width:100%;height:100%;object-fit:cover;display:block;">'
        if thumb else
        '<div style="width:100%;height:100%;display:flex;align-items:center;justify-content:center;font-size:32px;color:#1f2937;">📷</div>'
    )

    return (
        f'<div class="hist-card">'
        f'<div class="hist-thumb">'
        f'<div class="hist-thumb-inner">{thumb_img}</div>'
        f'<span class="hist-badge" style="background:{b_bg};color:{b_color};border:1px solid {b_border};">{badge}</span>'
        f'</div>'
        f'<div class="hist-body">'
        f'<div class="hist-date">📅 {date_str}</div>'
        f'<div class="hist-cap">{cap_short if cap_short else "<i style=\"color:#374151\">No caption</i>"}</div>'
        f'<div class="hist-stats"><span>❤️ {likes:,}</span><span>💬 {comments:,}</span>{extra}</div>'
        f'<a href="{html_module.escape(link)}" target="_blank" class="hist-view-link">View ↗</a>'
        f'</div>'
        f'</div>'
    )


_ROW = 5

def _render_grid(posts: list) -> None:
    if not posts:
        return
    for row_start in range(0, len(posts), _ROW):
        row = posts[row_start : row_start + _ROW]
        cols = st.columns(_ROW, gap="small")
        for col, post in zip(cols, row):
            with col:
                st.markdown(_card_html(post), unsafe_allow_html=True)
        if row_start + _ROW < len(posts):
            st.markdown(
                '<hr style="border:none;border-top:1px solid rgba(139,92,246,0.12);margin:10px 0 8px 0;">',
                unsafe_allow_html=True)


st.markdown("""
<div style="margin-bottom:18px;">
  <div style="font-size:22px;font-weight:700;color:#e2e8f0;margin-bottom:3px;">📋 Post History</div>
  <div style="font-size:13px;color:#6b7280;">All posts published through NewsFlow AI — live from your connected accounts</div>
</div>
""", unsafe_allow_html=True)

if not has_fb and not has_ig:
    st.error("No social platforms configured. Add FACEBOOK_ACCESS_TOKEN and platform IDs to your .env file, then check API Status Dashboard.")
    st.stop()

_ctrl_l, _ctrl_m, _ctrl_r = st.columns([2, 3, 1])

with _ctrl_l:
    platform_opts = []
    if has_ig: platform_opts.append("📸 Instagram")
    if has_fb: platform_opts.append("📘 Facebook")
    if len(platform_opts) > 1:
        platform_opts = ["All Platforms"] + platform_opts
    selected = st.selectbox("Platform", platform_opts, label_visibility="collapsed")

with _ctrl_m:
    limit = st.select_slider(
        "Posts to load", options=[6, 12, 24, 48], value=12,
        label_visibility="collapsed",
        format_func=lambda x: f"Last {x} posts",
    )

with _ctrl_r:
    st.button("🔄 Refresh", width='stretch')   # triggers a rerun

st.markdown('<hr style="border:none;border-top:1px solid rgba(139,92,246,0.1);margin:12px 0;">', unsafe_allow_html=True)

all_posts: list = []
errors: list[str] = []

with st.spinner("Fetching posts from connected accounts…"):
    if selected in ("All Platforms", "📸 Instagram") and has_ig:
        ig_posts, ig_err = fetch_instagram(limit)
        if ig_err:
            errors.append(f"📸 Instagram: {ig_err}")
        else:
            all_posts += ig_posts

    if selected in ("All Platforms", "📘 Facebook") and has_fb:
        fb_posts, fb_err = fetch_facebook(limit)
        if fb_err:
            errors.append(f"📘 Facebook: {fb_err}")
        else:
            all_posts += fb_posts

for err in errors:
    st.error(err)

all_posts.sort(key=lambda p: p.get("_sort_time", ""), reverse=True)

if all_posts:
    ig_count = sum(1 for p in all_posts if p.get("_platform") == "instagram")
    fb_count = sum(1 for p in all_posts if p.get("_platform") == "facebook")
    total_likes = sum(
        (p.get("like_count") or p.get("reactions", {}).get("summary", {}).get("total_count") or 0)
        for p in all_posts
    )
    total_comments = sum(
        (p.get("comments_count") or p.get("comments", {}).get("summary", {}).get("total_count") or 0)
        for p in all_posts
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Posts",     len(all_posts))
    m2.metric("📸 Instagram",    ig_count)
    m3.metric("📘 Facebook",     fb_count)
    m4.metric("Total Engagement", f"{total_likes + total_comments:,}")

    st.markdown(f'<div style="font-size:11px;color:#4a5568;margin:8px 0 14px 0;">'
                f'Showing {len(all_posts)} post(s) · sorted by newest first</div>',
                unsafe_allow_html=True)

    _render_grid(all_posts)

else:
    if not errors:
        st.markdown("""
<div style="text-align:center;padding:80px 20px;background:#0a0a14;
     border:1px dashed rgba(139,92,246,0.15);border-radius:12px;color:#4a5568;">
  <div style="font-size:52px;margin-bottom:14px;">📭</div>
  <div style="font-size:16px;font-weight:600;margin-bottom:6px;">No posts yet</div>
  <div style="font-size:12px;color:#374151;">
    Run the pipeline and publish content to see it here
  </div>
</div>""", unsafe_allow_html=True)
