import streamlit as st
import html as html_module
from dotenv import load_dotenv

load_dotenv(override=True)

st.set_page_config(
    page_title="API Status — NewsFlow AI",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [data-testid="stApp"] {
    background: #0d0d1a !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
}
[data-testid="stSidebar"] { background:#0a0a14 !important; }

/* hide default page nav */
[data-testid="stSidebarNavItems"],
section[data-testid="stSidebarNavSeparator"] { display:none !important; }

[data-testid="stButton"] button[kind="primary"] {
    background:linear-gradient(135deg,#7c3aed,#8b5cf6) !important;
    border:none !important;border-radius:8px !important;font-weight:600 !important;
    box-shadow:0 4px 15px rgba(139,92,246,0.4) !important;
}
[data-testid="stButton"] button[kind="secondary"] {
    background:rgba(139,92,246,0.08) !important;
    border:1px solid rgba(139,92,246,0.3) !important;
    color:#a78bfa !important;border-radius:8px !important;font-weight:500 !important;
}
[data-testid="stMetric"] { background:#13131f;border-radius:10px;padding:12px 16px;
    border:1px solid rgba(139,92,246,0.15); }
[data-testid="stMetricLabel"] { color:#6b7280 !important;font-size:11px !important; }
[data-testid="stMetricValue"] { color:#e2e8f0 !important; }
[data-testid="stProgress"] > div > div {
    background:linear-gradient(90deg,#7c3aed,#8b5cf6,#06b6d4) !important; }
hr { border-color:rgba(139,92,246,0.1) !important; }

.api-card { background:#13131f;border-radius:12px;padding:18px;
    border:1px solid rgba(255,255,255,0.06);border-left:3px solid;
    margin-bottom:14px;transition:transform 0.15s,box-shadow 0.15s; }
.api-card:hover { transform:translateY(-2px);box-shadow:0 8px 30px rgba(0,0,0,0.5); }
.api-header { display:flex;justify-content:space-between;align-items:center;margin-bottom:8px; }
.api-name-row { display:flex;align-items:center;gap:8px; }
.status-dot { width:10px;height:10px;border-radius:50%;display:inline-block;flex-shrink:0; }
.dot-active  { animation:pulse-g 2s ease-in-out infinite; }
.dot-limited { animation:pulse-y 1s ease-in-out infinite; }
@keyframes pulse-g  { 0%,100%{box-shadow:0 0 4px rgba(16,185,129,.3);}50%{box-shadow:0 0 12px rgba(16,185,129,.9);} }
@keyframes pulse-y  { 0%,100%{box-shadow:0 0 4px rgba(245,158,11,.3);}50%{box-shadow:0 0 12px rgba(245,158,11,.9);} }
.api-name { font-weight:700;font-size:15px;color:#e2e8f0; }
.api-icon { font-size:20px; }
.api-badge { font-size:10px;font-weight:700;padding:3px 10px;border-radius:20px;
    letter-spacing:.5px;text-transform:uppercase; }
.api-type  { font-size:10px;color:#6b7280;margin-bottom:6px;text-transform:uppercase;letter-spacing:.5px; }
.api-note  { font-size:12px;color:#94a3b8;margin-bottom:8px;line-height:1.4; }
.api-detail{ font-size:11px;color:#64748b;font-family:'JetBrains Mono',monospace;margin-top:4px; }
.api-key   { font-size:10px;color:#374151;font-family:'JetBrains Mono',monospace;margin-top:6px; }
.usage-wrap{ margin:8px 0 2px 0; }
.usage-bg  { background:#1e1e32;border-radius:4px;height:6px;overflow:hidden;margin-bottom:3px; }
.usage-fill{ height:100%;border-radius:4px;transition:width .8s ease; }
.usage-txt { font-size:10px;color:#64748b; }
.latency   { font-size:10px;color:#8b5cf6;margin-left:6px;font-family:'JetBrains Mono',monospace; }

.pipeline-banner {
    display:flex;align-items:center;gap:10px;
    background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.25);
    border-radius:8px;padding:8px 16px;margin-bottom:16px;font-size:12px;color:#f59e0b;
}
.pipeline-banner-done {
    background:rgba(16,185,129,0.08);border-color:rgba(16,185,129,0.25);color:#10b981;
}
.pipeline-banner-idle {
    background:rgba(100,116,139,0.08);border-color:rgba(100,116,139,0.2);color:#64748b;
}
.pulse-dot {
    width:8px;height:8px;border-radius:50%;flex-shrink:0;
    animation:pulse-y 1s ease-in-out infinite;background:#f59e0b;
}
</style>
""", unsafe_allow_html=True)

try:
    from shared_state import _ps, _ps_lock
    with _ps_lock:
        pipeline_running = _ps.get("running", False)
        pipeline_step    = _ps.get("step", 0)
        sub_text         = _ps.get("sub_text", "")
        pipeline_logs    = list(_ps.get("logs", []))
except Exception:
    pipeline_running = False
    pipeline_step    = 0
    sub_text         = ""
    pipeline_logs    = []

st.markdown("""
<style>
header[data-testid="stHeader"] { display:none !important; }
#MainMenu { display:none !important; }
[data-testid="stSidebarNavItems"],
section[data-testid="stSidebarNavSeparator"] { display:none !important; }
[data-testid="stSidebar"] { display:none !important; }
.main .block-container { padding-top:68px !important; }
.nf-navbar {
    position:fixed; top:0; left:0; right:0; z-index:999999;
    height:52px; background:#07070f;
    border-bottom:1px solid rgba(139,92,246,0.16);
    display:flex; align-items:center; padding:0 28px; gap:0;
    font-family:'Inter',-apple-system,sans-serif;
}
.nb-logo { display:flex; align-items:center; gap:9px; margin-right:28px; flex-shrink:0; }
.nb-logo-icon { font-size:20px; filter:drop-shadow(0 0 7px rgba(139,92,246,0.9)); }
.nb-logo-text { font-size:15px; font-weight:700; color:#e2e8f0; letter-spacing:-0.3px; white-space:nowrap; }
.nb-div  { width:1px; height:22px; background:rgba(139,92,246,0.18); margin-right:6px; flex-shrink:0; }
.nb-item {
    display:inline-flex; align-items:center; padding:0 16px; height:52px;
    font-size:13px; font-weight:500; color:#6b7280; cursor:pointer;
    border-bottom:2px solid transparent; transition:color 0.15s,border-color 0.15s; white-space:nowrap;
}
.nb-item:hover { color:#c4b5fd; }
.nb-active { color:#e2e8f0 !important; border-bottom-color:#8b5cf6 !important; font-weight:600 !important; }
</style>""", unsafe_allow_html=True)

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
    <span class="nb-item nb-active">API Status Dashboard</span>
    <span class="nb-item" onclick="window.location.href='/history'">History</span>
    <span class="nb-item" onclick="window.location.href='/rag_creator'" style="cursor:pointer;">RAG Creator</span>
  `;
  p.body.prepend(nav);
})();
</script>""", height=1)

col_hdr, col_refresh = st.columns([5, 1])
with col_hdr:
    st.markdown("""
<div style="display:flex;align-items:center;gap:12px;padding:4px 0;">
  <span style="font-size:28px;filter:drop-shadow(0 0 10px rgba(139,92,246,.7));">📡</span>
  <div>
    <div style="font-size:22px;font-weight:700;color:#e2e8f0;">API Status Dashboard</div>
    <div style="font-size:11px;color:#6b7280;">Live health check of every service powering NewsFlow AI</div>
  </div>
</div>""", unsafe_allow_html=True)
with col_refresh:
    refresh = st.button("🔄 Refresh", type="primary", width='stretch')

st.markdown("---")

agent_names = {1:"Trend Research",2:"Content Creation",3:"Image Generation",
               4:"Video Rendering",5:"Verification",6:"Publishing",7:"Publishing"}
if pipeline_running:
    agent = agent_names.get(pipeline_step, "Unknown")
    st.markdown(f"""
<div class="pipeline-banner">
  <div class="pulse-dot"></div>
  <b>Pipeline is running</b> — Agent {pipeline_step}: {agent}
  <span style="color:#94a3b8;margin-left:8px;">· {html_module.escape(sub_text)}</span>
  <span style="margin-left:auto;font-size:10px;color:#92400e;">
    This page does not affect the pipeline — it continues in the background.
  </span>
</div>""", unsafe_allow_html=True)
elif pipeline_step >= 6:
    st.markdown("""
<div class="pipeline-banner pipeline-banner-done">
  ✅ <b>Pipeline complete</b> — go back to approve and publish posts.
</div>""", unsafe_allow_html=True)
else:
    st.markdown("""
<div class="pipeline-banner pipeline-banner-idle">
  ⏸ Pipeline is idle — run it from the main page.
</div>""", unsafe_allow_html=True)


def _card(api: dict) -> str:
    cmap = {"active":"#10b981","degraded":"#f59e0b","rate_limited":"#f59e0b",
            "error":"#ef4444","not_configured":"#6b7280"}
    color   = cmap.get(api["status"], "#6b7280")
    dot_cls = {"active":"dot-active","degraded":"dot-limited",
               "rate_limited":"dot-limited"}.get(api["status"], "")

    usage_html = ""
    if api.get("usage_pct") is not None:
        pct = api["usage_pct"]
        bc  = "#10b981" if pct < 60 else "#f59e0b" if pct < 85 else "#ef4444"
        usage_html = f"""
<div class="usage-wrap">
  <div class="usage-bg"><div class="usage-fill" style="width:{pct}%;background:{bc};"></div></div>
  <span class="usage-txt">{html_module.escape(api.get('usage_text',''))}</span>
</div>"""

    key_html = f'<div class="api-key">🔑 {html_module.escape(api.get("key_preview",""))}</div>' \
               if api.get("key_preview") else ""
    lat  = f'<span class="latency">~{api.get("latency_ms","?")}ms</span>' \
           if api.get("latency_ms") else ""
    det  = f'<div class="api-detail">{html_module.escape(api.get("detail",""))}{lat}</div>' \
           if api.get("detail") else ""

    return f"""
<div class="api-card" style="border-left-color:{color};">
  <div class="api-header">
    <div class="api-name-row">
      <span class="status-dot {dot_cls}" style="background:{color};"></span>
      <span class="api-icon">{api['icon']}</span>
      <span class="api-name">{html_module.escape(api['name'])}</span>
    </div>
    <span class="api-badge"
      style="background:{color}1a;color:{color};border:1px solid {color}40;">
      {html_module.escape(api['status_text'])}
    </span>
  </div>
  <div class="api-type">{html_module.escape(api['type'])}</div>
  <div class="api-note">{html_module.escape(api.get('note',''))}</div>
  {key_html}
  {usage_html}
  {det}
  <div class="api-key" style="margin-top:8px;color:#1f2937;">
    Last checked: {api.get('checked_at','—')}
  </div>
</div>"""


@st.cache_data(ttl=60, show_spinner=False)
def fetch_statuses():
    from tools.api_status import get_all_statuses
    return get_all_statuses()

if refresh:
    st.cache_data.clear()

with st.spinner("Checking all 8 APIs concurrently…"):
    statuses = fetch_statuses()

active  = sum(1 for s in statuses if s["status"] in ("active","degraded"))
limited = sum(1 for s in statuses if s["status"] == "rate_limited")
errors  = sum(1 for s in statuses if s["status"] == "error")
uncfg   = sum(1 for s in statuses if s["status"] == "not_configured")

m1, m2, m3, m4 = st.columns(4)
m1.metric("✅ Active",          active,  delta=None)
m2.metric("⚠️ Rate Limited",   limited, delta=None)
m3.metric("❌ Error",          errors,  delta=None)
m4.metric("🔧 Not Configured", uncfg,   delta=None)
st.progress(active / max(len(statuses), 1),
            text=f"{active}/{len(statuses)} APIs operational")
st.markdown("---")

left, right = st.columns(2)
for i, api in enumerate(statuses):
    col = left if i % 2 == 0 else right
    with col:
        st.markdown(_card(api), unsafe_allow_html=True)

st.markdown("---")

st.markdown("#### 📋 Free Tier Limits Quick Reference")
rows = [
    ("🧠", "Groq",            "100K tokens/day (70B) · 14.4K tokens/min (8B) · Auto-fallback between models"),
    ("📰", "NewsAPI",         "100 requests/day · Resets midnight UTC · Developer plan"),
    ("🤗", "HuggingFace",     "FLUX.1 / SDXL require Pro ($9/mo) · SD 2.1 free · Openverse fallback auto-activates"),
    ("🌸", "Pollinations.ai", "Free · 1 concurrent request per shared IP · 402 = instant skip to Openverse"),
    ("🖼️", "Openverse",       "Free · No key needed · Millions of CC photos from Flickr, Wikipedia & more"),
    ("☁️", "Cloudinary",      "25 GB storage + 25 GB bandwidth/month · Required for Instagram Reel upload"),
    ("📘", "Facebook",        "Graph API free for page publishing · Access token expires every 60 days"),
    ("📸", "Instagram",       "Business Account required · Uses same token as Facebook · Reels via video URL"),
]
for icon, name, note in rows:
    col_i, col_n, col_d = st.columns([0.4, 1.2, 5])
    col_i.markdown(f"<div style='font-size:18px;padding:2px 0'>{icon}</div>", unsafe_allow_html=True)
    col_n.markdown(f"<div style='font-size:12px;font-weight:600;color:#8b5cf6;padding-top:2px'>{name}</div>", unsafe_allow_html=True)
    col_d.markdown(f"<div style='font-size:11px;color:#6b7280;padding-top:2px'>{note}</div>", unsafe_allow_html=True)

st.markdown("---")
st.caption("Results cached for 60 seconds · Click 🔄 Refresh to force a recheck · All checks run concurrently (8 threads)")

if pipeline_running:
    import time
    time.sleep(3)
    st.rerun()
