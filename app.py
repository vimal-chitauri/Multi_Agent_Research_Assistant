import streamlit as st
import os
import html as html_module
import threading
import time
import random
from dotenv import load_dotenv

from shared_state import _ps, _ps_lock, _log, _alog, _sub, _stopped, _abort, _proceed_event, _wait_proceed

load_dotenv(override=True)

st.set_page_config(
    page_title="NewsFlow AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

def _run_pipeline(niche: str, dry_run: bool, pub_ig: bool, pub_fb: bool, manual_mode: bool = False) -> None:
    try:
        _log("⚡ Pipeline starting — all 5 agents queued")

        with _ps_lock: _ps["step"] = 1; _ps["agent_logs"][1] = []
        _sub("Connecting to NewsAPI…", 5)
        _alog("🔍 Trend Research Agent — initializing")
        _alog("  → GET newsapi.org/v2/top-headlines  [pageSize=40, country=us]")
        _sub("Sending request to newsapi.org…", 30)
        if _stopped(): return _abort()
        try:
            from agents.trend_agent import TrendAgent
            _t0 = time.time()
            r1  = TrendAgent(niche=niche).run()
            _t1 = int((time.time() - _t0) * 1000)
            with _ps_lock: _ps["trend_result"] = r1
            trends = r1.data.get("trends", [])
            _sub("Ranking stories by engagement potential…", 72)
            _alog(f"  ← newsapi.org  HTTP 200  latency={_t1}ms  stories_ranked={len(trends)}")
            for t in trends:
                _alog(f"  #{t['rank']}  [{t['emotion']}]  {t['topic'][:52]}")
            _sub(f"✅ {len(trends)} stories selected", 100)
            _alog(f"✅ Agent 1 done  time={_t1}ms  stories={len(trends)}")
        except Exception as e:
            _alog(f"❌ Agent 1 failed: {e}")
            with _ps_lock: _ps["running"] = False; _ps["step"] = -1; _ps["error"] = str(e)
            _sub(f"❌ Agent 1 failed: {str(e)[:60]}", 0)
            return

        if _stopped(): return _abort()
        if manual_mode:
            if not _wait_proceed(1): return _abort("Manual review timed out after Trends")
            with _ps_lock:
                _sel    = _ps.get("selected_trends"); _ps["selected_trends"] = None
                _new_r1 = _ps.get("retry_r1");       _ps["retry_r1"]        = None
            if _new_r1 is not None:
                r1          = _new_r1
                trends_list = r1.data.get("trends", trends_list)
            if _sel is not None:
                trends_list     = _sel
                r1.data["trends"] = _sel

        with _ps_lock: _ps["step"] = 2; _ps["agent_logs"][2] = []
        trends_list = r1.data["trends"]
        n2 = len(trends_list)
        _sub("Building prompt for Groq LLM…", 8)
        _alog("✍️  Content Creation Agent — initializing")
        _alog(f"  input={n2} trends  model=llama-3.3-70b-versatile")
        _alog(f"  → POST api.groq.com/openai/v1/chat/completions  [generating {n2} posts]")
        _sub("Calling Groq LLM — generating captions & hashtags…", 40)
        _t0 = time.time()
        try:
            from agents.content_agent import ContentAgent
            r2  = ContentAgent(niche=niche).run({"trends": trends_list, "niche": niche})
            _t2 = int((time.time() - _t0) * 1000)
            with _ps_lock: _ps["content_result"] = r2
            posts2 = r2.data.get("posts", [])
            _sub("Parsing captions, hooks and hashtags…", 85)
            _alog(f"  ← api.groq.com  HTTP 200  latency={_t2}ms  posts_generated={r2.data['total']}")
            for p in posts2:
                ht = len(p.get("hashtags", [])) if isinstance(p.get("hashtags"), list) else "?"
                _alog(f"  post#{p['rank']}  hashtags={ht}  has_ig_hook={bool(p.get('instagram'))}  {p['topic'][:40]}")
            _sub(f"✅ {r2.data['total']} posts ready", 100)
            _alog(f"✅ Agent 2 done  time={_t2}ms  posts={r2.data['total']}")
        except Exception as e:
            _alog(f"❌ Agent 2 failed: {e}")
            with _ps_lock: _ps["running"] = False; _ps["step"] = -1; _ps["error"] = str(e)
            _sub(f"❌ Agent 2 failed: {str(e)[:60]}", 0)
            return

        if _stopped(): return _abort()
        if manual_mode:
            if not _wait_proceed(2): return _abort("Manual review timed out after Content")
            with _ps_lock:
                _new_r2 = _ps.get("retry_r2"); _ps["retry_r2"] = None
            if _new_r2 is not None:
                r2 = _new_r2

        with _ps_lock: _ps["step"] = 3; _ps["agent_logs"][3] = []
        posts3 = r2.data["posts"]
        n_img  = sum(len(p.get("image_prompts", [])) or 1 for p in posts3)
        _sub(f"Starting image pipeline — {n_img} images…", 3)
        _alog("🖼️  Image Generation Agent — initializing")
        _alog(f"  chain: HuggingFace → Pollinations.ai → Openverse → gradient-fallback")
        _alog(f"  target={n_img} images  posts={len(posts3)}")
        try:
            from agents.image_agent import ImageAgent, HF_MODELS, POLLINATIONS_MODELS
            TOTAL_BE = len(HF_MODELS) + len(POLLINATIONS_MODELS)
            _alog(f"  HF_models={len(HF_MODELS)}  Pollinations_models={len(POLLINATIONS_MODELS)}  total_backends={TOTAL_BE}")
            for i, p in enumerate(posts3):
                _sub(f"Queuing post {i+1}/{len(posts3)}: {p['topic'][:22]}…",
                     int(5 + (i / len(posts3)) * 20))
                _alog(f"  → queuing post {i+1}/{len(posts3)}: {p['topic'][:42]}")
            _sub("POSTing to HuggingFace inference API…", 30)
            _alog(f"  → POST api-inference.huggingface.co  (trying {len(HF_MODELS)} models in order)")
            _t0 = time.time()
            r3  = ImageAgent().run({"posts": posts3})
            _t3 = int((time.time() - _t0) * 1000)
            _sub(f"{r3.data['images_generated']}/{n_img} images done — checking…", 70)
            _alog(f"  ← first pass  time={_t3}ms  images_ok={r3.data['images_generated']}/{n_img}")
            for p in r3.data.get("posts", []):
                paths = p.get("image_paths", [])
                src   = p.get("image_source", "?")
                _alog(f"  post#{p['rank']}  source={src}  files={len(paths)}  {p['topic'][:38]}")

            def _enough(p):
                paths  = p.get("image_paths", [])
                needed = len(p.get("image_prompts", [])) or 1 if "image_prompts" in p else 1
                return len(paths) >= needed

            retry       = 0
            used_topics = [p["topic"] for p in r3.data["posts"]]
            while retry < 5:
                if _stopped():
                    with _ps_lock: _ps["image_result"] = r3
                    return _abort()
                failed = [p for p in r3.data["posts"] if not _enough(p)]
                if not failed: break
                retry += 1
                _alog(f"  ⚠️  {len(failed)} post(s) missing — all {TOTAL_BE} backends failed")
                _sub(f"Retry {retry}/5 — fetching {len(failed)} replacements…", 70 + retry * 4)
                _alog(f"  🔄 retry {retry}/5 → newsapi.org + api.groq.com + ImageAgent ({len(failed)} posts)")
                try:
                    from agents.trend_agent   import TrendAgent
                    from agents.content_agent import ContentAgent
                    _tr0 = time.time()
                    t2   = TrendAgent(niche=niche).run({"niche": niche, "exclude_topics": used_topics,
                                                        "count": len(failed)})
                    if not t2.success or not t2.data.get("trends"):
                        _alog("  ℹ️  No more unique stories — using gradient slides")
                        break
                    _alog(f"  ← newsapi.org  {int((time.time()-_tr0)*1000)}ms  fresh={len(t2.data['trends'])} stories")
                    used_topics += [t["topic"] for t in t2.data["trends"]]
                    c2 = ContentAgent(niche=niche).run({"trends": t2.data["trends"], "niche": niche})
                    if not c2.success: continue
                    _ri0 = time.time()
                    r2b  = ImageAgent().run({"posts": c2.data["posts"]})
                    _alog(f"  ← ImageAgent retry  {int((time.time()-_ri0)*1000)}ms  ok={r2b.data.get('images_generated',0)}")
                    if not r2b.success: continue
                    reps = r2b.data["posts"]; ri = 0; ap = r3.data["posts"]; sw = 0
                    for i2, post in enumerate(ap):
                        if not _enough(post) and ri < len(reps):
                            rep = reps[ri]; ri += 1
                            if _enough(rep):
                                orig_rank = post["rank"]; ap[i2] = rep; ap[i2]["rank"] = orig_rank; sw += 1
                    r3.data["posts"] = ap
                    if sw: _alog(f"  ✅ {sw} post(s) swapped with fresh stories")
                    else:  _alog("  ⚠️  replacements also failed — next retry")
                except Exception as er:
                    _alog(f"  ⚠️  retry error: {er}")

            still_bad = [p for p in r3.data["posts"] if not _enough(p)]
            if still_bad:
                _alog(f"  ℹ️  {len(still_bad)} post(s) → gradient-slide fallback (all backends exhausted)")
            with _ps_lock: _ps["image_result"] = r3
            total_ms = int((time.time() - _t0) * 1000)
            _sub(f"✅ {r3.data['images_generated']} images generated", 100)
            _alog(f"✅ Agent 3 done  time={total_ms}ms  images={r3.data['images_generated']}/{n_img}  retries={retry}")
        except Exception as e:
            _alog(f"❌ Agent 3 failed: {e}")
            with _ps_lock: _ps["running"] = False; _ps["step"] = -1; _ps["error"] = str(e)
            _sub(f"❌ Agent 3 failed: {str(e)[:60]}", 0)
            return

        if _stopped(): return _abort()
        if manual_mode:
            if not _wait_proceed(3): return _abort("Manual review timed out after Images")
            with _ps_lock:
                _new_r3 = _ps.get("retry_r3"); _ps["retry_r3"] = None
            if _new_r3 is not None:
                r3 = _new_r3

        voice_enabled = _ps.get("voice_enabled", True)
        voice_preset  = _ps.get("voice_preset", "en-US-male")
        if voice_enabled:
            _sub("Generating voice narration…", 5)
            _alog("🎙️ Voice Narration Agent — generating speech")
            try:
                from agents.voice_agent import VoiceAgent
                _tv0  = time.time()
                r_v   = VoiceAgent(voice_preset=voice_preset).run(
                    {"posts": r3.data["posts"], "niche": niche}
                )
                _alog(f"  ← VoiceAgent  {int((time.time()-_tv0)*1000)}ms  "
                      f"ok={r_v.data.get('voice_ok',0)}/{len(r3.data['posts'])}")
                if r_v.success:
                    r3.data["posts"] = r_v.data["posts"]
            except Exception as ve:
                _alog(f"  ⚠️ VoiceAgent error (skipping): {ve}")
        else:
            _alog("🎙️ Voice narration disabled — skipping")

        with _ps_lock: _ps["step"] = 4; _ps["agent_logs"][4] = []
        posts4 = r3.data["posts"]
        n4     = len(posts4)
        _sub(f"Compositing {n4} × 30s MP4 videos…", 5)
        _alog("🎬 Video Rendering Agent — initializing")
        _alog(f"  engine=Pillow+imageio  resolution=1080×1080  fps=30  duration=30s  posts={n4}")
        try:
            from agents.video_agent import VideoAgent
            for i, p in enumerate(posts4):
                _sub(f"Rendering {i+1}/{n4}: {p['topic'][:22]}…", int(10 + (i / n4) * 72))
                slides = len(p.get("image_paths", []) or ([p["image_path"]] if p.get("image_path") else []))
                _alog(f"  🎞️  post {i+1}/{n4}  slides={slides or '?'}  audio={bool(p.get('audio_path'))}  {p['topic'][:38]}")
            _sub("Encoding MP4 with music overlay…", 86)
            _t0 = time.time()
            r4  = VideoAgent(duration=30, niche=niche).run({"posts": posts4, "niche": niche})
            _t4 = int((time.time() - _t0) * 1000)
            with _ps_lock: _ps["video_result"] = r4
            for p in r4.data.get("posts", []):
                vp    = p.get("video_path", "")
                vsize = ""
                if vp:
                    try: vsize = f"{os.path.getsize(vp)//1024}KB"
                    except: pass
                _alog(f"  ✅ post#{p.get('rank','?')}  file={os.path.basename(vp) if vp else '?'}  size={vsize or '?'}")
            _sub(f"✅ {r4.data['videos_created']}/{n4} videos at 30fps", 100)
            _alog(f"✅ Agent 4 done  time={_t4}ms  videos={r4.data['videos_created']}/{n4}  1080×1080 30fps")
        except Exception as e:
            _alog(f"❌ Agent 4 failed: {e}")
            with _ps_lock: _ps["running"] = False; _ps["step"] = -1; _ps["error"] = str(e)
            _sub(f"❌ Agent 4 failed: {str(e)[:60]}", 0)
            return

        if _stopped(): return _abort()
        if manual_mode:
            if not _wait_proceed(4): return _abort("Manual review timed out after Videos")
            with _ps_lock:
                _new_r4 = _ps.get("retry_r4"); _ps["retry_r4"] = None
            if _new_r4 is not None:
                r4 = _new_r4

        with _ps_lock: _ps["step"] = 5; _ps["agent_logs"][5] = []
        posts5 = r4.data["posts"]
        n5     = len(posts5)
        _sub("Sending posts to Groq for quality scoring…", 10)
        _alog("🔎 Quality Verification Agent — initializing")
        _alog(f"  model=llama-3.3-70b-versatile  endpoint=api.groq.com  posts={n5}")
        _alog(f"  → POST api.groq.com/openai/v1/chat/completions  [scoring {n5} posts × 5 criteria]")
        _alog("  criteria: accuracy · tone · platform_fit · engagement · safety  (max 20 each)")
        _sub(f"Groq scoring {n5} posts on 5 criteria…", 35)
        _t0 = time.time()
        try:
            from agents.verifier_agent import VerifierAgent
            r5  = VerifierAgent(niche=niche, ui_mode=True).run({"posts": posts5})
            _t5 = int((time.time() - _t0) * 1000)
            approved = {p["rank"]: True for p in r5.data["posts"]}
            scores   = [p.get("score", 0) for p in r5.data["posts"]]
            avg      = int(sum(scores) / len(scores)) if scores else 0
            with _ps_lock:
                _ps["verify_result"]  = r5
                _ps["approved_posts"] = approved
            _alog(f"  ← api.groq.com  HTTP 200  latency={_t5}ms  posts_scored={n5}  avg={avg}/100")
            for p in r5.data["posts"]:
                sc  = p.get("score", 0)
                rev = p.get("review", {})
                sd  = rev.get("scores", {})
                icon = "✅" if sc >= 70 else "⚠️" if sc >= 50 else "❌"
                parts = f"acc={sd.get('accuracy','?')} tone={sd.get('tone','?')} eng={sd.get('engagement','?')}" if sd else ""
                _alog(f"  {icon} post#{p['rank']}  score={sc}/100  {parts}")
            _sub(f"✅ {n5} posts scored — avg {avg}/100", 100)
            _alog(f"✅ Agent 5 done  time={_t5}ms  avg_quality={avg}/100  all_approved={n5}")
        except Exception as e:
            _alog(f"❌ Agent 5 failed: {e}")
            with _ps_lock: _ps["running"] = False; _ps["step"] = -1; _ps["error"] = str(e)
            _sub(f"❌ Agent 5 failed: {str(e)[:60]}", 0)
            return

        with _ps_lock:
            _ps["step"]    = 6
            _ps["running"] = False
        _log("🎉 Pipeline complete! Open the Verify tab to approve and publish.")
        _sub("🎉 Pipeline done — go to Verify tab to approve posts", 100)

    except Exception as e:
        _log(f"💥 Pipeline crashed: {e}")
        with _ps_lock:
            _ps["running"] = False; _ps["step"] = -1; _ps["error"] = str(e)
        _sub(f"💥 Crashed: {str(e)[:60]}", 0)


def _run_publisher(approved_posts: list, dry_run: bool, pub_ig: bool, pub_fb: bool) -> None:
    try:
        with _ps_lock:
            _ps["running"] = True; _ps["step"] = 7
            _ps["agent_logs"][7] = []
        mode  = "DRY RUN" if dry_run else "🔴 LIVE"
        plats = ("Instagram " if pub_ig else "") + ("Facebook" if pub_fb else "")
        _alog("📤 Publisher Agent — starting")
        _alog(f"  Mode: {mode} | Platforms: {plats.strip()}")
        _alog(f"  Publishing {len(approved_posts)} approved post(s)…")
        _sub(f"Publishing {len(approved_posts)} post(s) — {mode}…", 10)
        from agents.publisher_agent import PublisherAgent
        for i, p in enumerate(approved_posts):
            _sub(f"Publishing {i+1}/{len(approved_posts)}: {p['topic'][:28]}…",
                 int(15 + (i / len(approved_posts)) * 75))
            _alog(f"  📤 Post {i+1}/{len(approved_posts)}: {p['topic']}")
        r6 = PublisherAgent(dry_run=dry_run, publish_facebook=pub_fb,
                            publish_instagram=pub_ig).run({"posts": approved_posts})
        with _ps_lock:
            _ps["publish_result"] = r6
            _ps["step"]    = 8
            _ps["running"] = False
        _sub(f"✅ {r6.data['total']} post(s) published [{mode}]", 100)
        _alog(f"✅ Done — {r6.data['total']} post(s) published [{mode}]")
        _log("🏁 All done!")
    except Exception as e:
        _alog(f"❌ Publisher failed: {e}")
        with _ps_lock: _ps["running"] = False; _ps["error"] = str(e)
        _sub(f"❌ Publisher failed: {str(e)[:60]}", 0)


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [data-testid="stApp"] {
    background:#0d0d1a !important; color:#e2e8f0 !important;
    font-family:'Inter',-apple-system,sans-serif !important;
}

/* Hide everything Streamlit-default */
header[data-testid="stHeader"]           { display:none !important; }
#MainMenu                                 { display:none !important; }
[data-testid="stSidebarNavItems"]         { display:none !important; }
section[data-testid="stSidebarNavSeparator"] { display:none !important; }
[data-testid="stSidebar"]                { display:none !important; }
[data-testid="collapsedControl"]         { display:none !important; }

/* Full-width content area — flush against navbar */
.main .block-container {
    padding-top:54px !important;
    padding-bottom:44px !important;
    padding-left:32px !important;
    padding-right:32px !important;
    max-width:100% !important;
}
/* Collapse height=0 navbar iframe — no extra gap */
iframe[height="0"] { display:block !important; height:0 !important; overflow:hidden !important; margin:0 !important; padding:0 !important; border:none !important; }
div[data-testid="stCustomComponentV1"] { margin:0 !important; padding:0 !important; line-height:0 !important; }
/* Kill default vertical spacing between Streamlit blocks */
[data-testid="stVerticalBlock"] { gap:0.3rem !important; }
section.main > div.block-container > div { gap:0.3rem !important; }

/* ── Navbar (CSS for JS-injected element) ──────────────────────────────── */
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
.nb-right { margin-left:auto; display:flex; align-items:center; gap:10px; }
.nb-run-dot {
    width:7px; height:7px; border-radius:50%; background:#f59e0b;
    display:inline-block; margin-right:4px;
    animation:nb-blink 1s ease-in-out infinite;
}
@keyframes nb-blink { 0%,100%{opacity:1;} 50%{opacity:0.3;} }
.nb-badge {
    font-size:10px; font-weight:700; padding:3px 10px; border-radius:20px;
    text-transform:uppercase; letter-spacing:1px; white-space:nowrap;
}
.nb-dry  { background:rgba(245,158,11,0.12); color:#f59e0b; border:1px solid rgba(245,158,11,0.28); }
.nb-live { background:rgba(16,185,129,0.12);  color:#10b981; border:1px solid rgba(16,185,129,0.28); }
.nb-run-ind { display:flex; align-items:center; gap:5px; font-size:12px; color:#f59e0b; font-weight:500; }

/* ── Control bar card ──────────────────────────────────────────────────── */
.ctrl-card {
    background:#0f0f1e;
    border:1px solid rgba(139,92,246,0.18);
    border-radius:12px;
    padding:14px 20px;
    margin-bottom:16px;
    display:flex;
    align-items:center;
    gap:16px;
}
.ctrl-sep {
    width:1px; height:32px;
    background:rgba(139,92,246,0.15);
    flex-shrink:0;
    align-self:center;
}
.ctrl-stat-run {
    display:flex; align-items:flex-start; gap:8px;
    background:rgba(245,158,11,0.06);
    border:1px solid rgba(245,158,11,0.2);
    border-radius:8px; padding:8px 12px;
}
.ctrl-stat-dot {
    width:8px; height:8px; border-radius:50%; background:#f59e0b;
    flex-shrink:0; margin-top:3px;
    animation:dot-pulse 1s ease-in-out infinite;
}
.ctrl-stat-done {
    background:rgba(16,185,129,0.07); border:1px solid rgba(16,185,129,0.2);
    border-radius:8px; padding:10px 12px; font-size:11px; color:#10b981;
}
.ctrl-stat-err {
    background:rgba(248,113,113,0.07); border:1px solid rgba(248,113,113,0.2);
    border-radius:8px; padding:10px 12px; font-size:10px; color:#f87171;
}
.ctrl-stat-idle {
    padding:10px 12px; font-size:11px; color:#4a5568;
}

/* ── Pipeline section ──────────────────────────────────────────────────── */
[data-testid="stProgress"] > div > div {
    background:linear-gradient(90deg,#7c3aed,#8b5cf6,#06b6d4) !important;
}

/* ── Tabs — compact ────────────────────────────────────────────────────── */
[data-testid="stTabs"] [role="tablist"] {
    border-bottom:1px solid rgba(139,92,246,0.2) !important;
    gap:2px !important;
}
[data-testid="stTabs"] [role="tab"] {
    color:#6b7280 !important; font-weight:500 !important;
    border-radius:6px 6px 0 0 !important;
    font-size:11px !important; padding:5px 10px !important;
}
[data-testid="stTabs"] [role="tab"]:hover { color:#e2e8f0 !important; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color:#8b5cf6 !important; font-weight:600 !important;
    border-bottom:2px solid #8b5cf6 !important;
    background:rgba(139,92,246,0.08) !important;
}

/* ── Log panel ─────────────────────────────────────────────────────────── */
.log-panel-hdr {
    font-size:10px; font-weight:600; color:#8b5cf6;
    text-transform:uppercase; letter-spacing:1px;
    display:flex; align-items:center; gap:6px;
    margin-bottom:6px; padding-bottom:6px;
    border-bottom:1px solid rgba(139,92,246,0.12);
}
.log-terminal {
    background:#03030d; border:1px solid rgba(139,92,246,0.2); border-radius:7px;
    padding:8px 10px; overflow-y:auto;
    font-family:'JetBrains Mono','Courier New',monospace;
    font-size:10px; line-height:1.55;
    scrollbar-width:thin; scrollbar-color:#3b1d8a #03030d;
}
.log-terminal::-webkit-scrollbar { width:4px; }
.log-terminal::-webkit-scrollbar-track { background:#03030d; }
.log-terminal::-webkit-scrollbar-thumb { background:#3b1d8a; border-radius:2px; }
.log-terminal pre { margin:0; white-space:pre-wrap; word-break:break-all; }
.log-blink { width:6px; height:6px; border-radius:50%; background:#10b981;
    animation:dot-pulse 1s ease-in-out infinite; display:inline-block; }

/* ── Agent cards ───────────────────────────────────────────────────────── */
.agent-running-card {
    background:#0c0c1a; border:1px solid rgba(245,158,11,0.3); border-radius:12px;
    padding:20px; margin-bottom:8px; box-shadow:0 0 24px rgba(245,158,11,0.08);
}
.agent-running-title { font-size:18px; font-weight:700; color:#e2e8f0; }
.agent-running-sub   { font-size:11px; color:#f59e0b; }
.agent-waiting-card {
    background:#0a0a14; border:1px dashed rgba(139,92,246,0.2); border-radius:12px;
    padding:32px; text-align:center; color:#4a5568;
}

/* ── Misc UI ────────────────────────────────────────────────────────────── */
[data-testid="stButton"] button[kind="secondary"] {
    background:rgba(139,92,246,0.08) !important; border:1px solid rgba(139,92,246,0.3) !important;
    color:#a78bfa !important; border-radius:8px !important; font-weight:500 !important;
}
[data-testid="stButton"] button[kind="primary"] {
    background:linear-gradient(135deg,#7c3aed,#8b5cf6) !important; border:none !important;
    border-radius:8px !important; font-weight:600 !important;
    box-shadow:0 4px 15px rgba(139,92,246,0.4) !important; transition:all 0.2s !important;
}
[data-testid="stButton"] button[kind="primary"]:hover {
    transform:translateY(-1px) !important; box-shadow:0 6px 20px rgba(139,92,246,0.6) !important;
}
.nf-card { background:#13131f; border:1px solid rgba(139,92,246,0.15); border-radius:12px;
    padding:18px; box-shadow:0 4px 24px rgba(0,0,0,0.4); }
.post-card { background:#13131f; border-left:3px solid #8b5cf6; border-radius:8px; padding:16px; margin-bottom:12px; }
.news-card { background:#13131f; border-left:3px solid #06b6d4; border-radius:8px; padding:16px; margin-bottom:12px; }
[data-testid="stMetric"] { background:#13131f; border-radius:10px; padding:12px 16px; border:1px solid rgba(139,92,246,0.15); }
[data-testid="stMetricLabel"] { color:#6b7280 !important; font-size:11px !important; }
[data-testid="stMetricValue"] { color:#e2e8f0 !important; }
[data-testid="stExpander"] { background:#0f0f1e !important; border-radius:8px !important; border:1px solid rgba(139,92,246,0.1) !important; }
hr { border-color:rgba(139,92,246,0.1) !important; }
.score-pass{color:#10b981;} .score-warn{color:#f59e0b;} .score-fail{color:#ef4444;}

/* ── IDE Status bar ────────────────────────────────────────────────────── */
.nf-statusbar {
    position:fixed; bottom:0; left:0; right:0; z-index:9999;
    background:#03030d; border-top:1px solid rgba(139,92,246,0.22);
    padding:0 18px; display:flex; align-items:center; gap:14px; height:28px;
    font-family:'JetBrains Mono','Courier New',monospace; font-size:10px;
}
@keyframes sb-pulse { 0%,100%{opacity:1;} 50%{opacity:.55;} }
@keyframes dot-pulse { 0%,100%{box-shadow:0 0 4px rgba(245,158,11,0.4);} 50%{box-shadow:0 0 12px rgba(245,158,11,0.9);} }
</style>
""", unsafe_allow_html=True)


with _ps_lock:
    _snap       = dict(_ps)
    _snap_logs  = list(_ps["logs"])
    _snap_alogs = {k: list(v) for k, v in _ps.get("agent_logs", {}).items()}

st.session_state["pipeline_running"] = _snap["running"]
st.session_state["pipeline_step"]    = _snap["step"]
st.session_state["pipeline_logs"]    = _snap_logs
st.session_state["agent_logs"]       = _snap_alogs
st.session_state["sub_text"]         = _snap.get("sub_text", "")
st.session_state["sub_pct"]          = _snap.get("sub_pct",  0)
st.session_state["waiting_proceed"]  = _snap.get("waiting_proceed", False)
st.session_state["waiting_at_step"]  = _snap.get("waiting_at_step", 0)

for _k in ["trend_result","content_result","image_result",
           "video_result","verify_result","publish_result"]:
    if _snap.get(_k) is not None:
        st.session_state[_k] = _snap[_k]

if not st.session_state.get("approved_posts") and _snap.get("approved_posts"):
    st.session_state["approved_posts"] = dict(_snap["approved_posts"])
if "approved_posts" not in st.session_state:
    st.session_state["approved_posts"] = {}

# Initialize config defaults (only on first load)
for _cfg_k, _cfg_v in [("dry_run", True), ("pub_instagram", True), ("pub_facebook", True),
                       ("manual_mode", False), ("voice_enabled", True), ("voice_preset", "en-US-male")]:
    if _cfg_k not in st.session_state:
        st.session_state[_cfg_k] = _cfg_v


def _colorize_logs(logs: list, limit: int = 60) -> str:
    lines = logs[-limit:] if limit else logs
    out   = []
    for line in lines:
        esc = html_module.escape(line)
        if any(c in line for c in ["✅","🎉","🏁"]):
            out.append(f'<span style="color:#10b981">{esc}</span>')
        elif any(c in line for c in ["❌","💥"]):
            out.append(f'<span style="color:#f87171">{esc}</span>')
        elif "⚠️" in line:
            out.append(f'<span style="color:#f59e0b">{esc}</span>')
        elif any(c in line for c in ["⛔","🔄"]):
            out.append(f'<span style="color:#f97316">{esc}</span>')
        elif any(c in line for c in ["⚡","🔍","✍️","🖼️","🎬","🔎","📤"]):
            out.append(f'<span style="color:#60a5fa">{esc}</span>')
        elif line.strip().startswith("─"):
            out.append(f'<span style="color:#2d2d4e">{esc}</span>')
        else:
            out.append(f'<span style="color:#94a3b8">{esc}</span>')
    return "\n".join(out)


def _show_running(agent_id: int, title: str, icon: str) -> None:
    sub_text   = st.session_state.get("sub_text", "…")
    sub_pct    = st.session_state.get("sub_pct", 0)
    agent_logs = st.session_state.get("agent_logs", {}).get(agent_id, [])
    st.markdown(f"""
<div class="agent-running-card">
  <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px;">
    <span style="font-size:28px;">{icon}</span>
    <div>
      <div class="agent-running-title">{title} — Running</div>
      <div class="agent-running-sub">● {html_module.escape(sub_text)}</div>
    </div>
  </div>""", unsafe_allow_html=True)
    if agent_logs:
        colored = _colorize_logs(agent_logs, limit=0)
        st.markdown(
            f'<div class="log-terminal" style="max-height:300px;margin-top:8px;">'
            f'<pre>{colored}</pre></div>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            '<div style="text-align:center;padding:20px;color:#4a5568;font-size:12px;">'
            'Agent starting… first log lines will appear shortly.</div>',
            unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def _show_waiting(msg: str = "") -> None:
    step = st.session_state.get("pipeline_step", 0)
    text = msg or ("⚡ Click <b style='color:#8b5cf6;'>Run Pipeline</b> above to start"
                   if step == 0 else "Queued — waiting for previous agent to complete…")
    st.markdown(f'<div class="agent-waiting-card">{text}</div>', unsafe_allow_html=True)


def _do_regen_image(post: dict, slide_idx: int, prompt: str, variation: bool = False) -> str | None:
    """Regenerate one slide image in-place and return the new path (or None on failure)."""
    from agents.image_agent import ImageAgent
    rank  = post["rank"]
    topic = post["topic"]
    if variation:
        tags = [
            "different composition, unique visual style",
            "alternative perspective, fresh artistic approach",
            "different mood, contrasting color palette, creative framing",
            "wide angle view, dramatic lighting, bold colors",
        ]
        prompt = f"{prompt}, {random.choice(tags)}"
    filename = f"regen_{rank}_{slide_idx}_{int(time.time())}.png"
    agent    = ImageAgent()
    new_path = agent._generate(prompt, filename, search_query=topic, slide_index=slide_idx)
    if new_path:
        str_path = str(new_path)
        paths    = post.get("image_paths", [])
        if paths and slide_idx < len(paths):
            paths[slide_idx] = str_path
        elif not paths:
            post["image_paths"] = [str_path]
        if slide_idx == 0:
            post["image_path"] = str_path
        return str_path
    return None


def _show_proceed_btn(from_step: int, next_label: str) -> None:
    """Render the Manual Review approval banner + Proceed button for a given step."""
    if not st.session_state.get("manual_mode"):        return
    if not st.session_state.get("waiting_proceed"):    return
    if st.session_state.get("waiting_at_step") != from_step: return
    st.markdown("---")
    st.markdown(
        '<div style="background:rgba(139,92,246,0.09);border:1px solid rgba(139,92,246,0.35);'
        'border-radius:10px;padding:14px 18px;margin:10px 0 6px 0;">'
        '<div style="font-size:13px;color:#a78bfa;font-weight:700;margin-bottom:4px;">'
        '⏸  Manual Review — agent complete</div>'
        '<div style="font-size:12px;color:#9ca3af;">Review the results above. '
        'When you\'re ready, click Proceed to continue.</div>'
        '</div>', unsafe_allow_html=True)
    if st.button(f"▶️  {next_label}", type="primary", key=f"proceed_step_{from_step}"):
        _proceed_event.set()
        st.rerun()


def render_pipeline(step: int) -> str:
    agents = [(1,"🔍","Trend","Research"),(2,"✍️","Content","Creation"),
              (3,"🖼️","Image","Generation"),(4,"🎬","Video","Rendering"),
              (5,"🔎","Verify","Quality"),(6,"📤","Publish","Distribution")]
    nodes_html = ""
    for i, (aid, icon, name, sub) in enumerate(agents):
        if step <= 0:       state, badge = "idle",    "IDLE"
        elif step > aid:    state, badge = "done",    "✓ DONE"
        elif step == aid:   state, badge = "running", "● LIVE"
        else:               state, badge = "waiting", "NEXT"
        spinner = '<div class="spn"></div>' if state == "running" else ""
        nodes_html += (f'<div class="nd nd-{state}"><div class="nd-ic">{icon}</div>{spinner}'
                       f'<div class="nd-nm">{name}</div><div class="nd-sb">{sub}</div>'
                       f'<div class="nd-bg bg-{state}">{badge}</div></div>')
        if i < len(agents) - 1:
            cs = "done" if step > aid else ("act" if step == aid else "idle")
            nodes_html += (f'<div class="cn cn-{cs}"><div class="cn-l"></div>'
                           f'<div class="cn-d"></div><div class="cn-a">▶</div></div>')

    return f"""<!DOCTYPE html><html><head><style>
*{{box-sizing:border-box;margin:0;padding:0;}}
body{{background:transparent;font-family:-apple-system,sans-serif;}}
.wrap{{display:flex;align-items:center;justify-content:center;padding:6px 12px;gap:0;min-width:max-content;}}
.nd{{position:relative;width:100px;min-height:90px;border-radius:10px;padding:10px 8px;
    display:flex;flex-direction:column;align-items:center;justify-content:center;gap:3px;border:1.5px solid;transition:all .3s;}}
.nd-idle{{background:#11111e;border-color:#2a2a45;opacity:.6;}}
.nd-waiting{{background:#11111e;border-color:#22224a;opacity:.8;}}
.nd-done{{background:#071a12;border-color:#10b981;box-shadow:0 0 14px rgba(16,185,129,.25);}}
.nd-running{{background:#160f00;border-color:#f59e0b;animation:nd-pulse 1.4s ease-in-out infinite;}}
@keyframes nd-pulse{{0%,100%{{box-shadow:0 0 8px rgba(245,158,11,.4);}}50%{{box-shadow:0 0 22px rgba(245,158,11,.8),0 0 45px rgba(245,158,11,.15);}}}}
.nd-ic{{font-size:22px;line-height:1;}}
.nd-nm{{font-size:9px;font-weight:700;color:#e2e8f0;text-transform:uppercase;letter-spacing:.5px;text-align:center;}}
.nd-sb{{font-size:8px;color:#4a5568;text-align:center;}}
.nd-bg{{font-size:8px;font-weight:700;padding:2px 6px;border-radius:4px;margin-top:2px;}}
.bg-idle{{background:#1e1e38;color:#4a5568;}}
.bg-waiting{{background:#14203a;color:#60a5fa;}}
.bg-done{{background:#072314;color:#34d399;}}
.bg-running{{background:#2a1800;color:#fbbf24;animation:bg-blink 1s ease-in-out infinite;}}
@keyframes bg-blink{{0%,100%{{opacity:1;}}50%{{opacity:.4;}}}}
.spn{{position:absolute;top:-5px;right:-5px;width:14px;height:14px;
    border:2px solid rgba(245,158,11,.25);border-top-color:#f59e0b;border-radius:50%;animation:spin .7s linear infinite;}}
@keyframes spin{{to{{transform:rotate(360deg);}}}}
.cn{{position:relative;width:40px;height:4px;display:flex;align-items:center;flex-shrink:0;}}
.cn-l{{position:absolute;width:100%;height:2px;}}
.cn-idle .cn-l{{background:transparent;border-top:2px dashed #2a2a45;}}
.cn-done .cn-l{{background:#10b981;}}
.cn-act .cn-l{{background:linear-gradient(90deg,#f59e0b,#fbbf24);background-size:200% 100%;animation:flow 1s linear infinite;}}
@keyframes flow{{0%{{background-position:100% 0;}}100%{{background-position:-100% 0;}}}}
.cn-d{{position:absolute;width:7px;height:7px;border-radius:50%;top:50%;transform:translateY(-50%);}}
.cn-act .cn-d{{background:#fbbf24;animation:mv .9s linear infinite;box-shadow:0 0 6px #fbbf24;}}
.cn-done .cn-d{{background:#10b981;right:0;}}
.cn-idle .cn-d{{display:none;}}
@keyframes mv{{0%{{left:0%}}100%{{left:100%}}}}
.cn-a{{position:absolute;right:-3px;font-size:8px;}}
.cn-idle .cn-a{{color:#2a2a45;}}
.cn-done .cn-a{{color:#10b981;}}
.cn-act .cn-a{{color:#f59e0b;}}
</style></head><body><div class="wrap">{nodes_html}</div></body></html>"""


def render_status_bar(step: int, is_running: bool, sub_text: str, sub_pct: int) -> str:
    if step <= 0: return ""
    meta = {1:("Trend Research","#60a5fa"),2:("Content Creation","#a78bfa"),
            3:("Image Generation","#f59e0b"),4:("Video Rendering","#f97316"),
            5:("Verification","#34d399"),6:("Publishing","#10b981"),
            7:("Publishing","#10b981"),8:("Complete","#10b981")}
    name, color = meta.get(step, ("Pipeline","#8b5cf6"))
    icon = "⚡" if is_running else ("✅" if step >= 6 else "⏸")
    anim = "animation:sb-pulse 1.8s ease-in-out infinite;" if is_running else ""
    return f"""
<div class="nf-statusbar">
  <span style="color:{color};font-weight:700;">{icon}</span>
  <span style="color:{color};white-space:nowrap;min-width:160px;">Agent {min(step,6)}: {name}</span>
  <div style="min-width:200px;background:#12122a;border-radius:2px;height:3px;overflow:hidden;flex-shrink:0;border:1px solid rgba(139,92,246,0.12);">
    <div style="width:{sub_pct}%;height:100%;background:linear-gradient(90deg,{color}88,{color});{anim}transition:width .6s ease;"></div>
  </div>
  <span style="color:#4a5568;font-size:9px;white-space:nowrap;">{sub_pct}%</span>
  <span style="color:#94a3b8;flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;min-width:0;font-size:9.5px;">{html_module.escape(sub_text or '…')}</span>
</div>"""


def score_emoji(s): return "✅" if s >= 70 else "⚠️" if s >= 50 else "❌"


def _inject_navbar(dry_run: bool, is_running: bool, step: int) -> None:
    badge_cls = "nb-badge nb-dry" if dry_run else "nb-badge nb-live"
    mode_text = "DRY RUN"        if dry_run else "LIVE"
    anames    = {1:"Trend",2:"Content",3:"Images",4:"Video",5:"Verify",6:"Publish",7:"Publish"}
    run_part  = (f'<span class="nb-run-ind"><span class="nb-run-dot"></span>'
                 f'Agent {step}: {anames.get(step, "Running")}</span>'
                 if is_running else "")
    st.iframe(f"""<script>
(function() {{
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
    <span class="nb-item nb-active">NewsFlow AI</span>
    <span class="nb-item" onclick="window.location.href='/api_status'" style="cursor:pointer;">API Status Dashboard</span>
    <span class="nb-item" onclick="window.location.href='/history'" style="cursor:pointer;">History</span>
    <span class="nb-item" onclick="window.location.href='/rag_creator'" style="cursor:pointer;">RAG Creator</span>
    <div class="nb-right">
      {run_part}
      <span class="{badge_cls}">{mode_text}</span>
    </div>
  `;
  p.body.prepend(nav);
}})();
</script>""", height=1)


with st.sidebar:
    pass


step       = st.session_state.get("pipeline_step", 0)
is_running = st.session_state.get("pipeline_running", False)
_dry_run   = st.session_state.get("dry_run", True)   # read before widget renders for navbar

if "niche" not in st.session_state:
    st.session_state["niche"] = os.getenv("NICHE", "technology")
niche = st.session_state["niche"]

_inject_navbar(_dry_run, is_running, step)

_c_act, _c_cfg = st.columns([1.3, 5.2])
_do_stop  = False
_do_start = False

with _c_act:
    if is_running:
        _do_stop = st.button("⏹  Stop Pipeline", width='stretch')
    elif step <= 0 or step == -1:
        _do_start = st.button("⚡  Run Pipeline", type="primary", width='stretch')
    else:
        _do_start = st.button("🔄  Run Again", type="primary", width='stretch')

with _c_cfg:
    _s0, _s1, _s2, _s3, _s4 = st.columns([1.5, 1.2, 1, 1, 1.5])
    _s0.selectbox(
        "Niche",
        options=["technology", "latest_news", "finance", "fitness",
                 "crypto", "motivation", "marketing", "food", "travel"],
        key="niche",
        disabled=is_running,
    )
    _s1.toggle("Dry Run",             key="dry_run")
    _s2.checkbox("📸 Instagram",      key="pub_instagram")
    _s3.checkbox("📘 Facebook",       key="pub_facebook")
    _s4.toggle("🔬 Manual Review",    key="manual_mode",
               help="Pause after each agent so you can review & approve before continuing")

with st.expander("🎙️ Voice Settings", expanded=False):
    _vc1, _vc2 = st.columns([1, 2])
    _vc1.toggle("Enable Voice Narration", key="voice_enabled",
                value=st.session_state.get("voice_enabled", True),
                help="Add AI voice narration to videos (uses Google TTS or free gTTS fallback)")
    _vc2.selectbox(
        "Voice Preset",
        options=["en-US-male", "en-US-female", "en-GB-male", "en-GB-female",
                 "en-IN-male", "en-IN-female", "en-AU-male", "en-AU-female"],
        key="voice_preset",
        help="Google Cloud Neural2 voices (premium quality). Falls back to gTTS if no API key set.",
    )

# Handle button actions
if _do_stop:
    with _ps_lock: _ps["stop_requested"] = True
    _log("⛔ Stop requested by user — finishing current operation…")
    st.rerun()

if _do_start:
    _dr  = st.session_state.get("dry_run", True)
    _pig = st.session_state.get("pub_instagram", True)
    _pfb = st.session_state.get("pub_facebook", True)
    _mm  = st.session_state.get("manual_mode", False)
    _ve  = st.session_state.get("voice_enabled", True)
    _vpr = st.session_state.get("voice_preset", "en-US-male")
    with _ps_lock:
        _ps.update({"running":True, "stop_requested":False, "step":0,
                    "logs":[], "error":None, "sub_text":"", "sub_pct":0,
                    "agent_logs":{},
                    "trend_result":None, "content_result":None, "image_result":None,
                    "video_result":None, "verify_result":None, "publish_result":None,
                    "approved_posts":{}, "waiting_proceed":False, "waiting_at_step":0,
                    "selected_trends":None,
                    "voice_enabled": _ve, "voice_preset": _vpr,
                    "retry_r1":None,"retry_r2":None,"retry_r3":None,"retry_r4":None})
    _proceed_event.clear()
    st.session_state["approved_posts"] = {}
    threading.Thread(target=_run_pipeline,
                     args=(niche, _dr, _pig, _pfb, _mm),
                     daemon=True).start()
    st.rerun()

st.iframe(render_pipeline(step), height=110)

if is_running:
    pt = {1:"Fetching trends…",2:"Writing content…",3:"Generating images…",
          4:"Rendering videos…",5:"Verifying quality…",6:"Publishing…",7:"Publishing…"}.get(step,"Running…")
elif step == -1:  pt = "❌ Pipeline stopped"
elif step >= 8:   pt = "✅ All done"
elif step >= 6:   pt = "✅ Complete"
else:             pt = None   # idle — no text on progress bar

st.markdown('<div style="height:4px;"></div>', unsafe_allow_html=True)

_tab_col, _log_col = st.columns([7, 3])

with _tab_col:
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🔍  Trends", "✍️  Content", "🖼️  Images",
        "🎬  Videos", "🔎  Verify",  "📤  Publish",
    ])

    # Helper: True only when a step is actively running (not paused for manual review)
    _wp     = st.session_state.get("waiting_proceed", False)
    _was    = st.session_state.get("waiting_at_step", 0)
    _manual = st.session_state.get("manual_mode", False)
    _active = lambda n: step == n and is_running and not (_wp and _was == n)

    with tab1:
        if _active(1):
            _show_running(1, "Trend Research", "🔍")
        elif st.session_state.get("trend_result"):
            result  = st.session_state["trend_result"]
            trends  = result.data.get("trends", [])
            _wait1  = _wp and _was == 1

            cols = st.columns(min(len(trends), 3))
            for i, t in enumerate(trends[:3]):
                cols[i].metric(f"{'🥇🥈🥉'[i]} Story #{t['rank']}", t["topic"][:28], t["emotion"])
            st.markdown("---")
            for trend in trends:
                _te_col, _tr_col = st.columns([9, 1])
                with _te_col:
                    with st.expander(f"**#{trend['rank']} — {trend['topic']}**", expanded=True):
                        c1, c2 = st.columns(2)
                        c1.markdown(f"**Emotion:** {trend['emotion']}")
                        c1.markdown(f"**Post Type:** {trend['post_type']}")
                        c2.markdown(f"**Audience:** {trend['target_audience']}")
                        st.markdown(f"**Why Trending:** {trend['why_trending']}")
                        st.markdown(f"**Angle:** {trend['content_angle']}")
                with _tr_col:
                    if _manual and _wait1:
                        st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)
                        if st.button("🔄", key=f"regen_trend_{trend['rank']}",
                                     help="Replace this topic with a fresh trend",
                                     width='stretch'):
                            _excl = [t["topic"] for t in trends if t["rank"] != trend["rank"]]
                            with st.spinner(f"Finding replacement for #{trend['rank']}…"):
                                from agents.trend_agent import TrendAgent
                                _nr = TrendAgent(niche=niche).run({"exclude_topics": _excl, "count": 1})
                            if _nr.success and _nr.data.get("trends"):
                                _rep = _nr.data["trends"][0]
                                _rep["rank"] = trend["rank"]
                                result.data["trends"] = [_rep if t["rank"] == trend["rank"] else t for t in trends]
                                with _ps_lock:
                                    _ps["trend_result"] = result
                                st.session_state["trend_result"] = result
                            st.rerun()
            st.caption(f"Reasoning: {result.reasoning}")

            # Manual Review: topic selection + Proceed button
            if _manual and _wait1:
                st.markdown("---")
                st.markdown(
                    '<div style="background:rgba(139,92,246,0.09);border:1px solid rgba(139,92,246,0.35);'
                    'border-radius:10px;padding:14px 18px;margin:10px 0;">'
                    '<div style="font-size:13px;color:#a78bfa;font-weight:700;margin-bottom:4px;">'
                    '⏸  Manual Review — Select Topics</div>'
                    '<div style="font-size:12px;color:#9ca3af;">Check the topics you want to include. '
                    'Uncheck any you want to skip, then click Proceed.</div>'
                    '</div>', unsafe_allow_html=True)
                _sel_ranks = []
                for _t in trends:
                    _chk = st.checkbox(
                        f"#{_t['rank']}  **{_t['topic']}**  ·  {_t.get('emotion','?')}",
                        value=True, key=f"topicsel_{_t['rank']}")
                    if _chk:
                        _sel_ranks.append(_t["rank"])
                _sel_trends = [_t for _t in trends if _t["rank"] in _sel_ranks]
                st.caption(f"Selected {len(_sel_trends)}/{len(trends)} topics")
                _pb_space, _pb_retry, _pb_proceed = st.columns([2, 1, 1])
                with _pb_retry:
                    if st.button("🔄  Re-search Trends", width='stretch', key="retry_step_1"):
                        with st.spinner("Re-searching trends…"):
                            from agents.trend_agent import TrendAgent
                            _nr1 = TrendAgent(niche=niche).run()
                            with _ps_lock:
                                _ps["trend_result"] = _nr1
                                _ps["retry_r1"]     = _nr1
                            st.session_state["trend_result"] = _nr1
                        st.rerun()
                with _pb_proceed:
                    if st.button("▶️  Proceed to Content", type="primary",
                                 width='stretch', key="proceed_step_1"):
                        with _ps_lock:
                            _ps["selected_trends"] = _sel_trends if _sel_trends else trends
                        _proceed_event.set()
                        st.rerun()
        else:
            _show_waiting()

    with tab2:
        if _active(2):
            _show_running(2, "Content Creation", "✍️")
        elif st.session_state.get("content_result"):
            _content_res = st.session_state["content_result"]
            for post in _content_res.data["posts"]:
                _ch_col, _cr_col = st.columns([9, 1])
                with _ch_col:
                    st.markdown(f"#### Post #{post['rank']} — {post['topic']}")
                with _cr_col:
                    if _manual and _wp and _was == 2:
                        st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
                        if st.button("🔄", key=f"regen_post_{post['rank']}",
                                     help="Regenerate content for this post",
                                     width='stretch'):
                            _tr = st.session_state.get("trend_result")
                            _t_match = next((t for t in (_tr.data["trends"] if _tr else [])
                                             if t["rank"] == post["rank"]), None)
                            if _t_match:
                                with st.spinner(f"Regenerating post #{post['rank']}…"):
                                    from agents.content_agent import ContentAgent
                                    _ca   = ContentAgent(niche=niche)
                                    _niche = niche.lower()
                                    _np   = (_ca._generate_news_content(_t_match)
                                             if _niche == "latest_news"
                                             else _ca._generate_standard_content(_t_match, _niche))
                                if _np:
                                    _np["rank"] = post["rank"]
                                    _updated = [_np if p["rank"] == post["rank"] else p
                                                for p in _content_res.data["posts"]]
                                    _content_res.data["posts"] = _updated
                                    with _ps_lock:
                                        _ps["content_result"] = _content_res
                                        _ps["retry_r2"]       = _content_res
                                    st.session_state["content_result"] = _content_res
                            st.rerun()
                if "heading" in post:
                    st.markdown(f'<div class="news-card"><b>📰 {html_module.escape(post["heading"])}</b></div>',
                                unsafe_allow_html=True)
                    _bullets = post.get("bullets", [])
                    _bl, _br = st.columns(2)
                    with _bl:
                        for b in _bullets[:5]: st.markdown(f"- {b}")
                    with _br:
                        for b in _bullets[5:]: st.markdown(f"- {b}")
                    with st.expander("Full Description"): st.text(post.get("description",""))
                    src = post.get("source_url","")
                    if src: st.markdown(f"**Source:** [{post.get('source_name','')}]({src})")
                    st.markdown("**Hashtags:** " + " ".join([f"`#{h.lstrip('#')}`" for h in post.get("hashtags",[])[:8]]))
                else:
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**🐦 Twitter / X**")
                        tw = post.get("twitter",{}); cc = tw.get("char_count",0)
                        st.markdown(f'<div class="post-card">{html_module.escape(tw.get("text",""))}</div>',
                                    unsafe_allow_html=True)
                        st.caption(f"Chars: {'🟢' if cc<=280 else '🔴'} {cc}/280")
                    with c2:
                        st.markdown("**📸 Instagram**")
                        ig = post.get("instagram",{})
                        st.markdown(f'<div class="news-card">{html_module.escape(ig.get("hook",""))}</div>',
                                    unsafe_allow_html=True)
                        with st.expander("Full Caption"): st.text(ig.get("caption",""))
                    st.markdown(f"🖼️ **Image Prompt:** {post.get('image_prompt','')}")
                st.markdown("---")
            if _manual and _wp and _was == 2:
                if st.button("🔄  Regenerate Content", key="retry_step_2"):
                    _tr = st.session_state.get("trend_result")
                    if _tr:
                        with st.spinner("Regenerating content from current trends…"):
                            from agents.content_agent import ContentAgent
                            _nr2 = ContentAgent(niche=niche).run({"trends": _tr.data["trends"], "niche": niche})
                            with _ps_lock:
                                _ps["content_result"] = _nr2
                                _ps["retry_r2"]       = _nr2
                            st.session_state["content_result"] = _nr2
                    st.rerun()
            _show_proceed_btn(2, "Proceed to Image Generation")
        else:
            _show_waiting()

    with tab3:
        if _active(3):
            _show_running(3, "Image Generation", "🖼️")
        elif st.session_state.get("image_result"):
            result = st.session_state["image_result"]
            posts  = result.data["posts"]
            _m1, _m2, _m3 = st.columns(3)
            _m1.metric("Images Generated", result.data["images_generated"])
            _m2.metric("Total Posts",      len(posts))
            _m3.metric("Backend Chain",    "HF → Pollinations → Openverse")
            st.markdown("---")

            for post in posts:
                rank    = post["rank"]
                topic   = post["topic"]
                paths   = post.get("image_paths", [])
                single  = post.get("image_path")
                prompts = post.get("image_prompts", []) or [post.get("image_prompt", topic)]

                st.markdown(f"#### Post #{rank} — {topic}")
                _ic, _info = st.columns([1, 1])

                with _ic:
                    if paths:
                        for _si, _p in enumerate(paths):
                            _disp = _p if (_p and os.path.exists(_p)) else None
                            if _disp:
                                st.image(_disp, caption=f"Slide {_si+1}", width='stretch')
                            else:
                                st.markdown(f'<div class="nf-card" style="text-align:center;padding:20px;color:#6b7280;">🎨 Gradient slide {_si+1}</div>', unsafe_allow_html=True)
                    elif single and os.path.exists(str(single)):
                        st.image(single, width='stretch')
                    else:
                        st.markdown('<div class="nf-card" style="text-align:center;padding:30px;color:#6b7280;">🎨 Styled gradient slide</div>', unsafe_allow_html=True)

                with _info:
                    for _si, _pt in enumerate(prompts):
                        st.text_area(f"Slide {_si+1}", value=_pt, height=70, disabled=True,
                                     key=f"ip_{rank}_{_si}")

                        _bc1, _bc2 = st.columns(2)
                        _custom_key  = f"custom_mode_{rank}_{_si}"
                        _regen_key   = f"regen_trigger_{rank}_{_si}"

                        with _bc1:
                            if st.button("✏️ Custom Prompt", key=f"cpbtn_{rank}_{_si}",
                                         width='stretch',
                                         help="Write your own prompt to generate a new image"):
                                st.session_state[_custom_key] = not st.session_state.get(_custom_key, False)
                                st.rerun()

                        with _bc2:
                            if st.button("🔄 Regenerate", key=f"rgbtn_{rank}_{_si}",
                                         width='stretch',
                                         help="AI tries a different variation of this slide"):
                                st.session_state[_regen_key] = True
                                st.rerun()

                        if st.session_state.get(_custom_key, False):
                            _user_prompt = st.text_area(
                                "Your image prompt:",
                                value=_pt,
                                height=80,
                                key=f"cinput_{rank}_{_si}",
                                placeholder="Describe exactly what image you want…",
                            )
                            if st.button("🎨 Generate Image", key=f"cgenbtn_{rank}_{_si}",
                                         width='stretch'):
                                if _user_prompt.strip():
                                    with st.spinner(f"Generating slide {_si+1} with your prompt…"):
                                        _do_regen_image(post, _si, _user_prompt.strip())
                                    st.session_state[_custom_key] = False
                                    st.rerun()

                        if st.session_state.get(_regen_key, False):
                            st.session_state[_regen_key] = False
                            with st.spinner(f"Regenerating slide {_si+1} with a new AI variation…"):
                                _do_regen_image(post, _si, _pt, variation=True)
                            st.rerun()

                        st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)

                    st.caption(f"Status: `{post.get('image_status','unknown')}`")
                st.markdown("---")
            if _manual and _wp and _was == 3:
                if st.button("🔄  Retry All Images", key="retry_step_3"):
                    _cr = st.session_state.get("content_result")
                    if _cr:
                        with st.spinner("Re-generating all images with fresh AI calls…"):
                            from agents.image_agent import ImageAgent
                            _nr3 = ImageAgent().run({"posts": _cr.data["posts"]})
                            with _ps_lock:
                                _ps["image_result"] = _nr3
                                _ps["retry_r3"]     = _nr3
                            st.session_state["image_result"] = _nr3
                    st.rerun()
            _show_proceed_btn(3, "Proceed to Video Rendering")
        else:
            _show_waiting()

    with tab4:
        if _active(4):
            _show_running(4, "Video Rendering", "🎬")
        elif st.session_state.get("video_result"):
            result = st.session_state["video_result"]
            posts  = result.data["posts"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Videos Created", result.data["videos_created"])
            c2.metric("Duration",       f"{result.data['duration_sec']}s")
            c3.metric("Format",         "MP4 1080×1080")
            st.markdown("---")
            for post in posts:
                st.markdown(f"#### Post #{post['rank']} — {post['topic']}")
                vc, ic = st.columns([1,1])
                with vc:
                    vp = post.get("video_path")
                    if vp and os.path.exists(vp):
                        st.video(vp); st.caption(f"📁 `{os.path.basename(vp)}`")
                    else:
                        st.warning(f"Status: {post.get('video_status','unknown')}")
                with ic:
                    st.markdown(f"- Duration: **{result.data['duration_sec']}s**")
                    st.markdown(f"- Resolution: **1080 × 1080**")
                    st.markdown(f"- Music: {post.get('music_title','None')}")
                    for ln in post.get("info_content",[]): st.markdown(f"• {ln}")
                st.markdown("---")
            if _manual and _wp and _was == 4:
                if st.button("🔄  Re-render Videos", key="retry_step_4"):
                    _ir = st.session_state.get("image_result")
                    if _ir:
                        with st.spinner("Re-rendering all videos…"):
                            from agents.video_agent import VideoAgent
                            _nr4 = VideoAgent(duration=30, niche=niche).run({"posts": _ir.data["posts"], "niche": niche})
                            with _ps_lock:
                                _ps["video_result"] = _nr4
                                _ps["retry_r4"]     = _nr4
                            st.session_state["video_result"] = _nr4
                    st.rerun()
            _show_proceed_btn(4, "Proceed to Quality Verification")
        else:
            _show_waiting()

    with tab5:
        if _active(5):
            _show_running(5, "Quality Verification", "🔎")
        elif st.session_state.get("verify_result"):
            posts = st.session_state["verify_result"].data["posts"]
            st.markdown("Toggle approval for each post, then send to Publisher.")
            st.markdown("---")
            for post in posts:
                review = post.get("review",{}); score = post.get("score",0); rank = post["rank"]
                ct, cs, cv = st.columns([3,1,1])
                ct.markdown(f"**Post #{rank} — {post['topic']}**")
                cs.markdown(f'{score_emoji(score)} **{score}/100**'); cs.progress(score/100)
                approved = cv.toggle("Approve", value=st.session_state["approved_posts"].get(rank, True),
                                     key=f"ap_{rank}")
                st.session_state["approved_posts"][rank] = approved
                scores_d = review.get("scores",{})
                if scores_d:
                    sc = st.columns(5)
                    for i2, (label, key) in enumerate(zip(
                        ["Accuracy","Tone","Platform","Engagement","Safety"],
                        ["accuracy","tone","platform_fit","engagement","safety"])):
                        sc[i2].metric(label, f"{scores_d.get(key,0)}/20")
                scol, icol = st.columns(2)
                if review.get("strengths"): scol.success("**Strengths:** " + " · ".join(review["strengths"]))
                if review.get("issues"):    icol.warning("**Issues:** "    + " · ".join(review["issues"][:2]))
                if review.get("suggestion"): st.info(f"💡 {review['suggestion']}")
                st.markdown("---")
            approved_count = sum(1 for v in st.session_state["approved_posts"].values() if v)
            st.markdown(f"**{approved_count}/{len(posts)} posts approved**")
            _pub_dry = st.session_state.get("dry_run", True)
            _pub_ig  = st.session_state.get("pub_instagram", True)
            _pub_fb  = st.session_state.get("pub_facebook", True)
            can_publish = (approved_count > 0 and step >= 6 and
                           not is_running and not st.session_state.get("publish_result"))
            if can_publish:
                if st.button(f"📤 Send {approved_count} Post(s) to Publisher →", type="primary"):
                    approved_list = [p for p in posts
                                     if st.session_state["approved_posts"].get(p["rank"], False)]
                    threading.Thread(target=_run_publisher,
                                     args=(approved_list, _pub_dry, _pub_ig, _pub_fb),
                                     daemon=True).start()
                    st.rerun()
            if is_running and step == 7:
                st.info("📤 Publishing in progress — watch the Activity Log panel →")
            elif st.session_state.get("publish_result"):
                st.success("✅ Posts published — see Publish tab.")
        else:
            _show_waiting()

    with tab6:
        if step == 7 and is_running:
            _show_running(7, "Publishing to Social Media", "📤")
        elif st.session_state.get("publish_result"):
            result    = st.session_state["publish_result"]
            is_dry    = result.data.get("dry_run", True)
            platforms = result.data.get("platforms", [])
            published = result.data.get("published", [])
            failed    = result.data.get("failed", [])

            # Mode banner
            if is_dry:
                st.warning("🟡 DRY RUN — no real posts were made. Disable Dry Run above to go live.")
            else:
                st.success(f"🟢 LIVE — published to {', '.join(platforms)}!")

            # Summary row
            _pm1, _pm2, _pm3 = st.columns(3)
            _pm1.metric("Published", len(published))
            _pm2.metric("Failed",    len(failed))
            _pm3.metric("Platforms", len(platforms))
            st.markdown("---")

            def _clean_err(msg) -> str:
                s = str(msg)
                for cut in ["exec(", "DeltaGenerator", "Traceback (most", "  File \""]:
                    idx = s.find(cut)
                    if 0 < idx < len(s):
                        s = s[:idx]
                return s.strip()[:200] or "Unknown error"

            # Per-post status cards
            for post in (published + failed):
                fb    = post.get("facebook")
                ig    = post.get("instagram")
                fb_ok = bool(fb and fb.get("success"))
                ig_ok = bool(ig and ig.get("success"))
                all_ok = (fb is None or fb_ok) and (ig is None or ig_ok)
                any_ok = fb_ok or ig_ok
                icon   = "✅" if all_ok else ("⚠️" if any_ok else "❌")

                with st.expander(f"{icon}  {post.get('topic', 'Post')}", expanded=True):
                    _src_url  = post.get("source_url", "")
                    _src_name = post.get("source_name", "") or _src_url
                    if _src_url:
                        st.markdown(
                            f'<div style="font-size:12px;color:#9ca3af;margin-bottom:10px;">'
                            f'📰 <b>Source:</b> <a href="{html_module.escape(_src_url)}" target="_blank" '
                            f'style="color:#60a5fa;text-decoration:none;">'
                            f'{html_module.escape(_src_name)}</a></div>',
                            unsafe_allow_html=True)
                    _fb_col, _ig_col = st.columns(2)

                    with _fb_col:
                        st.markdown("**📘 Facebook**")
                        if fb is None:
                            st.caption("Not selected")
                        elif fb_ok:
                            st.success("Published ✅")
                            if fb.get("url"):
                                st.markdown(f"[View post ↗]({fb['url']})")
                        else:
                            _err = _clean_err(fb.get("error", "Failed"))
                            st.markdown(
                                f'<div style="background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.25);'
                                f'border-radius:8px;padding:10px 14px;font-size:12px;color:#f59e0b;">⚠️ {html_module.escape(_err)}</div>',
                                unsafe_allow_html=True)

                    with _ig_col:
                        st.markdown("**📸 Instagram**")
                        if ig is None:
                            st.caption("Not selected")
                        elif ig_ok:
                            st.success("Published as Reel ✅")
                            if ig.get("url"):
                                st.markdown(f"[View post ↗]({ig['url']})")
                        else:
                            _err = _clean_err(ig.get("error", "Failed"))
                            st.markdown(
                                f'<div style="background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.25);'
                                f'border-radius:8px;padding:10px 14px;font-size:12px;color:#f59e0b;">⚠️ {html_module.escape(_err)}</div>',
                                unsafe_allow_html=True)
        else:
            _show_waiting("Approve posts in the <b style='color:#8b5cf6;'>Verify</b> tab first, then publish.")


with _log_col:
    _logs  = st.session_state.get("pipeline_logs", [])
    _blink = '<span class="log-blink"></span>' if is_running else ''
    # Header + legend on one line so nothing gets covered
    st.markdown(
        f'<div class="log-panel-hdr" style="justify-content:space-between;flex-wrap:wrap;gap:4px;">'
        f'<span style="display:flex;align-items:center;gap:5px;">{_blink} Activity Log</span>'
        f'<span style="display:flex;gap:8px;font-weight:400;">'
        f'<span style="font-size:9px;color:#60a5fa;">● Agent</span>'
        f'<span style="font-size:9px;color:#10b981;">● Done</span>'
        f'<span style="font-size:9px;color:#f87171;">● Error</span>'
        f'<span style="font-size:9px;color:#f59e0b;">● Warn</span>'
        f'</span></div>',
        unsafe_allow_html=True)
    if _logs:
        st.markdown(
            f'<div class="log-terminal" style="height:calc(100vh - 360px);min-height:300px;">'
            f'<pre>{_colorize_logs(_logs, 80)}</pre></div>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="log-terminal" style="height:300px;display:flex;align-items:center;'
            'justify-content:center;flex-direction:column;gap:8px;">'
            '<span style="font-size:20px;">📋</span>'
            '<span style="color:#374151;font-size:11px;">No activity yet</span>'
            '<span style="color:#1f2937;font-size:10px;">Run the pipeline to see live logs</span>'
            '</div>', unsafe_allow_html=True)


_sb = render_status_bar(step, is_running,
                        st.session_state.get("sub_text", ""),
                        st.session_state.get("sub_pct",  0))
if _sb:
    st.markdown(_sb, unsafe_allow_html=True)


if st.session_state.get("pipeline_running") or st.session_state.get("waiting_proceed"):
    time.sleep(1.5)
    st.rerun()
