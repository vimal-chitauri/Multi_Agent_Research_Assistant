import json
import time
from datetime import datetime
from pathlib import Path

RAG_DB_DIR        = Path(__file__).parent.parent / "rag_db"
RAG_DB_DIR.mkdir(exist_ok=True)
SOCIAL_COLLECTION = "social_posts"
EXPERT_COLLECTION = "expert_knowledge"



def _get_ef():
    from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
    return ONNXMiniLM_L6_V2()


def _get_client():
    import chromadb
    return chromadb.PersistentClient(path=str(RAG_DB_DIR))


def _open_or_recreate(client, name: str, ef):
    try:
        return client.get_collection(name, embedding_function=ef)
    except Exception as e:
        if "conflict" in str(e).lower() or "mismatch" in str(e).lower():
            try:
                old = client.get_collection(name)
                data = old.get(include=["documents", "metadatas"])
                client.delete_collection(name)
                col = client.create_collection(name, embedding_function=ef,
                                               metadata={"hnsw:space": "cosine"})
                if data["ids"]:
                    col.add(ids=data["ids"], documents=data["documents"],
                            metadatas=data["metadatas"])
                return col
            except Exception:
                pass
        raise


def get_social_collection():
    ef     = _get_ef()
    client = _get_client()
    try:
        return _open_or_recreate(client, SOCIAL_COLLECTION, ef)
    except Exception:
        return client.create_collection(SOCIAL_COLLECTION, embedding_function=ef,
                                        metadata={"hnsw:space": "cosine"})


def get_expert_collection():
    ef     = _get_ef()
    client = _get_client()
    try:
        col = _open_or_recreate(client, EXPERT_COLLECTION, ef)
        if col.count() == 0:
            _seed_expert_knowledge(col)
        return col
    except Exception:
        col = client.create_collection(EXPERT_COLLECTION, embedding_function=ef,
                                       metadata={"hnsw:space": "cosine"})
        _seed_expert_knowledge(col)
        return col



def _blend_weights(social_count: int) -> tuple[float, float]:
    if social_count < 20:
        return 0.90, 0.10
    if social_count < 50:
        return 0.60, 0.40
    if social_count < 100:
        return 0.30, 0.70
    return 0.10, 0.90



def build_content_context(topic: str, niche: str) -> tuple[str, int]:
    try:
        social_col = get_social_collection()
        expert_col = get_expert_collection()
    except Exception:
        return "", 0

    social_count = social_col.count()
    ew, sw       = _blend_weights(social_count)
    query_text   = f"{topic} {niche}"

    expert_docs  = []
    expert_metas = []
    try:
        er = expert_col.query(
            query_texts=[query_text],
            n_results=min(20, expert_col.count()),
            include=["documents", "metadatas", "distances"],
        )
        expert_docs  = er["documents"][0]
        expert_metas = er["metadatas"][0]
    except Exception:
        pass

    own_docs  = []
    own_metas = []
    if social_count > 0:
        try:
            sr = social_col.query(
                query_texts=[topic],
                n_results=min(5, social_count),
                include=["documents", "metadatas", "distances"],
            )
            own_docs  = sr["documents"][0]
            own_metas = sr["metadatas"][0]
        except Exception:
            pass

    def _by_cat(cat, limit=3):
        return [d for d, m in zip(expert_docs, expert_metas)
                if m.get("category") == cat][:limit]

    hooks      = _by_cat("hook", 3)
    framework  = _by_cat("framework", 1)
    triggers   = _by_cat("trigger", 2)
    ctas       = _by_cat("cta", 1)
    hashtags   = _by_cat("hashtag", 1)
    style_tips = _by_cat("style", 2)
    formats    = _by_cat("format", 1)

    lines = []

    if framework:
        lines.append("CAPTION STRUCTURE TO FOLLOW:")
        lines.append(framework[0])
        lines.append("")

    if hooks:
        lines.append("PROVEN SCROLL-STOPPING HOOKS (adapt one as your opening line):")
        for h in hooks:
            lines.append(f"• {h}")
        lines.append("")

    if triggers:
        lines.append("ENGAGEMENT TECHNIQUES TO USE:")
        for t in triggers:
            lines.append(f"• {t}")
        lines.append("")

    if style_tips:
        lines.append("WRITING STYLE RULES:")
        for s in style_tips:
            lines.append(f"• {s}")
        lines.append("")

    if formats:
        lines.append("VIRAL FORMAT PATTERN:")
        lines.append(formats[0])
        lines.append("")

    if hashtags:
        lines.append("HASHTAG STRATEGY:")
        lines.append(hashtags[0])
        lines.append("")

    if ctas:
        lines.append("CALL TO ACTION (end with one of these):")
        lines.append(ctas[0])
        lines.append("")

    if own_docs and sw > 0:
        paired = sorted(zip(own_docs, own_metas),
                        key=lambda x: x[1].get("engagement", 0), reverse=True)
        top_own = paired[:3]
        if top_own:
            label = ("YOUR BEST POSTS — match this exact voice and style:"
                     if social_count >= 20 else
                     "YOUR POSTS SO FAR — start matching this voice:")
            lines.append(label)
            for doc, meta in top_own:
                eng = meta.get("engagement", 0)
                lines.append(f"[{eng} engagement] {doc[:300]}")
                lines.append("")

    lines.append(f"[Account has {social_count} posts — apply expert guidance at {int(ew*100)}% weight]")

    context = "\n".join(lines).strip()
    return context, social_count



def index_published_post(post: dict, result: dict) -> bool:
    try:
        platform = result.get("platform", "unknown")
        post_id  = result.get("post_id", "")

        if "description" in post and post.get("description"):
            caption = post["description"]
            src     = post.get("source_url", "")
            if src:
                caption += f"\n\nSource: {post.get('source_name','')}\n{src}"
        else:
            caption = post.get("instagram", {}).get("caption", "")

        caption = (caption or "").strip()
        if not caption:
            return False

        hashtags = post.get("hashtags", [])
        if not hashtags and "instagram" in post:
            hashtags = post["instagram"].get("hashtags", [])

        doc_id = (f"{platform}_{post_id}"
                  if post_id else
                  f"{platform}_{post.get('rank',0)}_{int(time.time())}")

        col = get_social_collection()
        col.upsert(
            ids=[doc_id],
            documents=[caption],
            metadatas=[{
                "platform":    platform,
                "post_id":     post_id,
                "likes":       0,
                "comments":    0,
                "engagement":  0,
                "has_image":   int(bool(post.get("image_path") or post.get("image_paths"))),
                "image_url":   "",
                "permalink":   result.get("url", "")[:500],
                "timestamp":   datetime.now().isoformat()[:20],
                "hashtags":    json.dumps([f"#{t.lstrip('#')}" for t in hashtags][:20]),
                "caption_len": len(caption),
                "topic":       post.get("topic", ""),
                "niche":       post.get("niche", ""),
            }],
        )
        return True
    except Exception as e:
        print(f"  [RAG] index_published_post failed: {e}")
        return False



def _seed_expert_knowledge(col) -> None:

    entries = []


    for i, (doc, emotion) in enumerate([
        ("Breaking: this just happened — and it is bigger than the media is telling you", "urgency"),
        ("If you have not heard about this yet, read this now. It is important.", "urgency"),
        ("The real story behind what happened that nobody is talking about", "curiosity"),
        ("Scientists just announced something that changes everything we thought we knew", "curiosity"),
        ("World leaders are scrambling after this news. Here is what it means for you", "urgency"),
        ("Most people have no idea this is happening right now. Here is the full picture", "curiosity"),
        ("This just changed and it affects every single one of us. Here is why", "urgency"),
        ("We just got confirmation on something huge. This is not getting enough coverage", "shock"),
        ("The story that will define this year just started. Here is what you need to know", "urgency"),
        ("Nobody is covering this properly. Here is the full truth about what happened", "curiosity"),
    ], 1):
        entries.append({
            "id": f"hook_news_{i:03d}",
            "document": doc,
            "metadata": {"category": "hook", "niche": "latest_news",
                         "emotion": emotion, "performance_tier": "A"}
        })

    for i, doc in enumerate([
        ("Scientists just discovered something that rewrites what we thought we knew about this topic", ),
        ("New research reveals something shocking that most experts did not expect to find",),
        ("For the first time in history, researchers have achieved something extraordinary",),
        ("The science behind this is more fascinating than anyone ever told you",),
        ("A new study just changed our understanding of this completely. Here is what they found",),
    ], 1):
        entries.append({
            "id": f"hook_science_{i:03d}",
            "document": doc[0],
            "metadata": {"category": "hook", "niche": "latest_news",
                         "emotion": "curiosity", "performance_tier": "A"}
        })

    for i, doc in enumerate([
        ("This new technology is about to change everything. Here is why it matters",),
        ("The tech behind this explained in plain English. It is more powerful than you think",),
        ("Most people do not understand how significant this technology actually is. Let me explain",),
        ("Scientists built something that was not supposed to be possible. Here is how",),
    ], 1):
        entries.append({
            "id": f"hook_tech_{i:03d}",
            "document": doc[0],
            "metadata": {"category": "hook", "niche": "technology",
                         "emotion": "curiosity", "performance_tier": "A"}
        })

    for i, doc in enumerate([
        ("What wealthy people know about this that they do not teach in school",),
        ("This just happened in the markets. Here is what smart investors are doing right now",),
        ("The financial opportunity most people are about to miss. Here is what is happening",),
        ("Why this matters more to your money than anything else happening right now",),
    ], 1):
        entries.append({
            "id": f"hook_finance_{i:03d}",
            "document": doc[0],
            "metadata": {"category": "hook", "niche": "finance",
                         "emotion": "urgency", "performance_tier": "A"}
        })

    for i, doc in enumerate([
        ("The mindset shift that changes everything. Most people never figure this out",),
        ("Stop doing it the old way. Here is what actually works",),
        ("What successful people know about this that average people never learn",),
        ("This simple change will transform how you think about your goals",),
    ], 1):
        entries.append({
            "id": f"hook_motivation_{i:03d}",
            "document": doc[0],
            "metadata": {"category": "hook", "niche": "motivation",
                         "emotion": "inspiration", "performance_tier": "A"}
        })

    for i, doc in enumerate([
        ("What 3 months of consistent training does to your body according to science",),
        ("The fitness truth that most trainers never tell you about this",),
        ("Why everything you know about this might be wrong. New research says otherwise",),
        ("This one change made a bigger difference than years of the wrong approach",),
    ], 1):
        entries.append({
            "id": f"hook_fitness_{i:03d}",
            "document": doc[0],
            "metadata": {"category": "hook", "niche": "fitness",
                         "emotion": "curiosity", "performance_tier": "A"}
        })


    entries.append({
        "id": "framework_news_001",
        "document": (
            "NEWS CAPTION FRAMEWORK: "
            "1. HOOK — one punchy shocking sentence that stops the scroll. "
            "2. WHAT HAPPENED — explain the event in 2-3 short sentences, plain English. "
            "3. WHY IT MATTERS — connect it to the reader's real life in 1-2 sentences. "
            "4. BIGGER PICTURE — what does this mean globally or long-term, 1-2 sentences. "
            "5. CTA — end with one specific engaging question. "
            "Use short paragraphs. One idea per line. Write like you are texting a smart friend."
        ),
        "metadata": {"category": "framework", "niche": "latest_news",
                     "emotion": "urgency", "performance_tier": "A"}
    })

    entries.append({
        "id": "framework_science_001",
        "document": (
            "SCIENCE CAPTION FRAMEWORK: "
            "1. SURPRISING FACT — lead with the most mind-blowing aspect of the discovery. "
            "2. PLAIN ENGLISH — explain what it means without any jargon, like talking to a curious 14-year-old. "
            "3. REAL WORLD — how does this affect everyday life or the future? Be specific. "
            "4. ZOOM OUT — what does this mean for humanity, medicine, or our understanding of the universe? "
            "5. WONDER — end with a thought-provoking question that makes them think. "
            "Make complex things feel exciting and accessible."
        ),
        "metadata": {"category": "framework", "niche": "latest_news",
                     "emotion": "curiosity", "performance_tier": "A"}
    })

    entries.append({
        "id": "framework_tech_001",
        "document": (
            "TECHNOLOGY CAPTION FRAMEWORK: "
            "1. BOLD CLAIM — state the breakthrough in one punchy line. "
            "2. THE PROBLEM — what existed before that needed solving? "
            "3. THE SOLUTION — what changed and how does it work, simply explained. "
            "4. PRACTICAL IMPACT — what does this mean for normal people in daily life? "
            "5. WHERE THIS LEADS — one sentence on the future this enables. "
            "6. CTA — ask for their opinion or prediction."
        ),
        "metadata": {"category": "framework", "niche": "technology",
                     "emotion": "curiosity", "performance_tier": "A"}
    })

    entries.append({
        "id": "framework_finance_001",
        "document": (
            "FINANCE CAPTION FRAMEWORK: "
            "1. HOOK — a statistic or fact that creates immediate financial urgency. "
            "2. CONTEXT — what is happening in the market or economy right now. "
            "3. IMPACT — what this means specifically for regular people and their money. "
            "4. ACTION — what informed people are doing about it. "
            "5. HONEST TAKE — your clear, confident perspective. No hedging. "
            "6. CTA — ask what they think or invite them to follow for more."
        ),
        "metadata": {"category": "framework", "niche": "finance",
                     "emotion": "urgency", "performance_tier": "A"}
    })

    entries.append({
        "id": "framework_motivation_001",
        "document": (
            "MOTIVATION CAPTION FRAMEWORK: "
            "1. PATTERN INTERRUPT — open with something unexpected, bold, or counter-intuitive. "
            "2. RELATABLE STRUGGLE — name the problem readers face honestly. "
            "3. THE SHIFT — what changes when you see it differently. "
            "4. PRACTICAL TRUTH — one actionable insight they can use today. "
            "5. MOMENTUM CLOSER — one short sentence that fires them up. "
            "6. CTA — simple action or question. Keep sentences short. Use white space."
        ),
        "metadata": {"category": "framework", "niche": "motivation",
                     "emotion": "inspiration", "performance_tier": "A"}
    })

    entries.append({
        "id": "framework_fitness_001",
        "document": (
            "FITNESS CAPTION FRAMEWORK: "
            "1. BOLD HEALTH CLAIM — state the result or discovery upfront. "
            "2. SCIENCE BACKING — cite a study, stat, or expert without being boring. "
            "3. PRACTICAL STEPS — 3-5 specific actionable points, each on its own line. "
            "4. WHAT TO EXPECT — timeline for results, be realistic and honest. "
            "5. COMMON MISTAKE — what most people get wrong about this. "
            "6. CLOSER — one motivating sentence that energises them. "
            "7. CTA — specific question about their experience."
        ),
        "metadata": {"category": "framework", "niche": "fitness",
                     "emotion": "inspiration", "performance_tier": "A"}
    })

    # ── ENGAGEMENT TRIGGERS ────────────────────────────────────────────────────

    entries.append({
        "id": "trigger_curiosity_001",
        "document": (
            "CURIOSITY GAP TRIGGER: State the fascinating conclusion first, then make "
            "them read to understand how you got there. Example opener: "
            "'The answer surprised even the researchers themselves. Here is what they found.' "
            "Creates irresistible pull to keep reading."
        ),
        "metadata": {"category": "trigger", "niche": "all",
                     "emotion": "curiosity", "performance_tier": "A"}
    })

    entries.append({
        "id": "trigger_question_001",
        "document": (
            "QUESTION ENDING TRIGGER: Always end with ONE specific question, not a vague one. "
            "BAD: 'What do you think?' "
            "GOOD: 'If this is true, what does it mean for how we approach this going forward? Comment below.' "
            "Specific questions get 3x more replies than generic ones. "
            "The question should be answerable in 1-2 sentences by anyone reading."
        ),
        "metadata": {"category": "trigger", "niche": "all",
                     "emotion": "curiosity", "performance_tier": "A"}
    })

    entries.append({
        "id": "trigger_save_001",
        "document": (
            "SAVE TRIGGER: Explicitly tell readers to save the post. "
            "Use: 'Save this post — there is more coming on this story' or "
            "'Bookmark this. You will want to refer back to it.' "
            "Posts with save calls-to-action get boosted by the Instagram algorithm "
            "as high-value content. Place it naturally near the end, before the question CTA."
        ),
        "metadata": {"category": "trigger", "niche": "all",
                     "emotion": "curiosity", "performance_tier": "A"}
    })

    entries.append({
        "id": "trigger_contrarian_001",
        "document": (
            "CONTRARIAN TRIGGER: Challenge conventional wisdom with something true but counter-intuitive. "
            "Start with 'Unpopular opinion:' or 'Most people have this completely backwards.' "
            "This drives comments from both sides — agreement and disagreement — "
            "and both types of engagement boost the algorithm equally. "
            "Make sure your contrarian take is defensible with a fact or study."
        ),
        "metadata": {"category": "trigger", "niche": "all",
                     "emotion": "shock", "performance_tier": "A"}
    })

    entries.append({
        "id": "trigger_social_proof_001",
        "document": (
            "SOCIAL PROOF TRIGGER: Reference scale to create FOMO and credibility. "
            "Examples: 'Over 50 million people are now...', '9 out of 10 experts agree...', "
            "'A study of 100,000 people found...'. "
            "Makes individual readers feel they are accessing collective wisdom. "
            "Be specific with numbers — '34 million' feels more real than 'millions'."
        ),
        "metadata": {"category": "trigger", "niche": "all",
                     "emotion": "urgency", "performance_tier": "A"}
    })

    entries.append({
        "id": "trigger_tag_001",
        "document": (
            "TAG TRIGGER: 'Tag someone who needs to see this' dramatically increases reach. "
            "Use it when content has clear relevance to a specific type of person. "
            "GOOD: 'Tag a friend who has been following this story.' "
            "GOOD: 'Tag someone who works in this field — they need to know this.' "
            "Only use when genuinely relevant, not on every post."
        ),
        "metadata": {"category": "trigger", "niche": "all",
                     "emotion": "empathy", "performance_tier": "B"}
    })

    entries.append({
        "id": "trigger_urgency_001",
        "document": (
            "URGENCY TRIGGER: Use time-relevant language to drive immediate engagement. "
            "Instead of timeless framing, say: 'This is happening right now', "
            "'As of today this changed', 'This just came out and here is why it matters'. "
            "Creates urgency that drives engagement in the first hour of posting, "
            "which is the most important window for the Instagram algorithm."
        ),
        "metadata": {"category": "trigger", "niche": "latest_news",
                     "emotion": "urgency", "performance_tier": "A"}
    })

    entries.append({
        "id": "trigger_relatable_001",
        "document": (
            "RELATABLE OPENER TRIGGER: Connect the big story to everyday life immediately. "
            "Example: 'You might not think this affects you. But it already does. Here is how.' "
            "Makes global or complex stories feel personal, which drives comments "
            "about personal experience — the highest-quality engagement signal."
        ),
        "metadata": {"category": "trigger", "niche": "all",
                     "emotion": "empathy", "performance_tier": "A"}
    })

    entries.append({
        "id": "trigger_pause_001",
        "document": (
            "PAUSE TRIGGER: End with a short philosophical statement that lingers. "
            "Examples: 'Think about that for a second.' / 'Let that sink in.' / "
            "'Take a moment with that.' "
            "Makes readers pause emotionally before scrolling. "
            "Increases time-on-post which the algorithm measures as quality signal."
        ),
        "metadata": {"category": "trigger", "niche": "motivation",
                     "emotion": "inspiration", "performance_tier": "B"}
    })

    entries.append({
        "id": "trigger_fomo_001",
        "document": (
            "FOMO TRIGGER: Create fear of missing out by framing content as exclusive insight. "
            "Examples: 'Most people will scroll past this and miss what matters.' "
            "'The people who understand this now will be ahead of everyone else.' "
            "'By the time this is mainstream news, it will be too late to act on it.' "
            "Creates a sense that reading is actively valuable, not passive."
        ),
        "metadata": {"category": "trigger", "niche": "finance",
                     "emotion": "urgency", "performance_tier": "A"}
    })

    # ── HASHTAG STRATEGIES ─────────────────────────────────────────────────────

    entries.append({
        "id": "hashtag_news_001",
        "document": (
            "HASHTAG STRATEGY FOR NEWS NICHE: Use 11-13 hashtags total. "
            "Formula: 3 mega (1M+ posts) + 5 mid (100K-1M) + 3 niche (10K-100K) + 2 topic-specific. "
            "Mega: #news #breakingnews #worldnews. "
            "Mid: #newsupdate #currentevents #latestnews #dailynews #globalnews. "
            "Niche: #newsoftheday #informative #newsstory. "
            "Topic-specific: use keywords directly from your headline as hashtags."
        ),
        "metadata": {"category": "hashtag", "niche": "latest_news",
                     "emotion": "neutral", "performance_tier": "A"}
    })

    entries.append({
        "id": "hashtag_tech_001",
        "document": (
            "HASHTAG STRATEGY FOR TECH NICHE: Use 11-13 hashtags. "
            "Mega: #technology #tech #innovation. "
            "Mid: #technews #futuretech #techupdate #artificialintelligence #digitalworld. "
            "Niche: #techinnovation #technologynews #futureoftech. "
            "Topic-specific: the actual technology name and directly related terms."
        ),
        "metadata": {"category": "hashtag", "niche": "technology",
                     "emotion": "neutral", "performance_tier": "A"}
    })

    entries.append({
        "id": "hashtag_finance_001",
        "document": (
            "HASHTAG STRATEGY FOR FINANCE NICHE: Use 10-12 hashtags. "
            "Mega: #finance #money #investing. "
            "Mid: #financialtips #moneytips #investing101 #personalfinance #wealthbuilding. "
            "Niche: #financialfreedom #moneyadvice #financialliteracy. "
            "Topic-specific: the specific financial event, instrument, or strategy."
        ),
        "metadata": {"category": "hashtag", "niche": "finance",
                     "emotion": "neutral", "performance_tier": "A"}
    })

    entries.append({
        "id": "hashtag_fitness_001",
        "document": (
            "HASHTAG STRATEGY FOR FITNESS NICHE: Use 10-12 hashtags. "
            "Mega: #fitness #health #workout. "
            "Mid: #fitnessmotivation #healthylifestyle #fitlife #gym #wellness. "
            "Niche: #fitnesstips #healthtips #workoutmotivation. "
            "Topic-specific: specific exercise, muscle group, or nutrition term."
        ),
        "metadata": {"category": "hashtag", "niche": "fitness",
                     "emotion": "neutral", "performance_tier": "A"}
    })

    entries.append({
        "id": "hashtag_motivation_001",
        "document": (
            "HASHTAG STRATEGY FOR MOTIVATION NICHE: Use 10-12 hashtags. "
            "Mega: #motivation #mindset #success. "
            "Mid: #motivationalquotes #successmindset #personaldevelopment #growthmindset #inspiration. "
            "Niche: #motivationmonday #successquotes #motivationdaily. "
            "Topic-specific: the specific concept, habit, or goal being discussed."
        ),
        "metadata": {"category": "hashtag", "niche": "motivation",
                     "emotion": "neutral", "performance_tier": "A"}
    })

    entries.append({
        "id": "hashtag_crypto_001",
        "document": (
            "HASHTAG STRATEGY FOR CRYPTO NICHE: Use 10-12 hashtags. "
            "Mega: #crypto #bitcoin #cryptocurrency. "
            "Mid: #cryptonews #blockchain #cryptotrading #bitcoinnews #altcoins. "
            "Niche: #cryptoupdate #cryptomarket #cryptocommunity. "
            "Topic-specific: specific coin name, protocol, or market event."
        ),
        "metadata": {"category": "hashtag", "niche": "crypto",
                     "emotion": "neutral", "performance_tier": "A"}
    })

    entries.append({
        "id": "hashtag_food_001",
        "document": (
            "HASHTAG STRATEGY FOR FOOD NICHE: Use 10-12 hashtags. "
            "Mega: #food #foodie #recipe. "
            "Mid: #foodphotography #instafood #homemade #cooking #healthyfood. "
            "Niche: #foodlover #eatinghealthy #foodblogger. "
            "Topic-specific: specific dish name, cuisine type, or key ingredient."
        ),
        "metadata": {"category": "hashtag", "niche": "food",
                     "emotion": "neutral", "performance_tier": "A"}
    })

    entries.append({
        "id": "hashtag_travel_001",
        "document": (
            "HASHTAG STRATEGY FOR TRAVEL NICHE: Use 10-12 hashtags. "
            "Mega: #travel #wanderlust #travelgram. "
            "Mid: #travelphotography #instatravel #traveltheworld #explore #adventure. "
            "Niche: #traveldiaries #travelblogger #travellife. "
            "Topic-specific: destination name, country, or type of travel experience."
        ),
        "metadata": {"category": "hashtag", "niche": "travel",
                     "emotion": "neutral", "performance_tier": "A"}
    })

    # ── CTA TEMPLATES ──────────────────────────────────────────────────────────

    entries.append({
        "id": "cta_comment_001",
        "document": (
            "COMMENT CTA TEMPLATES (use one): "
            "'What is your take on this? Drop it below.' "
            "'Agree or disagree? Let me know in the comments.' "
            "'Did you already know about this? Comment yes or no.' "
            "'Drop a fire emoji if this surprised you.' "
            "'One word to describe your reaction — comment below.' "
            "Always use a specific prompt, never just 'Comment below' alone."
        ),
        "metadata": {"category": "cta", "niche": "all",
                     "emotion": "curiosity", "performance_tier": "A"}
    })

    entries.append({
        "id": "cta_follow_001",
        "document": (
            "FOLLOW CTA TEMPLATES (use one naturally): "
            "'Follow for daily stories like this.' "
            "'Hit follow if you want to stay ahead of what is happening.' "
            "'We post the stories that matter, every single day.' "
            "Place follow CTA in the middle of the caption or at the end, "
            "never as the very first thing. Earn the follow, then ask for it."
        ),
        "metadata": {"category": "cta", "niche": "all",
                     "emotion": "neutral", "performance_tier": "A"}
    })

    entries.append({
        "id": "cta_save_001",
        "document": (
            "SAVE CTA TEMPLATES: "
            "'Save this post — you will want to refer back to it.' "
            "'Bookmark this for later.' "
            "'Save this and share it with someone who needs to see it.' "
            "The save action is the strongest engagement signal on Instagram. "
            "It tells the algorithm this content has long-term value."
        ),
        "metadata": {"category": "cta", "niche": "all",
                     "emotion": "neutral", "performance_tier": "A"}
    })

    entries.append({
        "id": "cta_share_001",
        "document": (
            "SHARE CTA TEMPLATES: "
            "'Share this with someone who would find this useful.' "
            "'Tag a friend who needs to know about this.' "
            "'Send this to one person right now — they will thank you.' "
            "Shares are the highest-reach action. "
            "Use share CTAs for content that is genuinely useful or surprising."
        ),
        "metadata": {"category": "cta", "niche": "all",
                     "emotion": "empathy", "performance_tier": "A"}
    })

    # ── WRITING STYLE TIPS ─────────────────────────────────────────────────────

    entries.append({
        "id": "style_linebreak_001",
        "document": (
            "LINE BREAK RULE: Use short paragraphs of 1-3 sentences maximum. "
            "Hit enter after every distinct thought. "
            "This creates white space that makes posts scannable on mobile screens. "
            "Long walls of text lose readers after line 2. "
            "Think: newspaper subheadings, not essay paragraphs. "
            "Every line break is a micro-hook keeping them reading."
        ),
        "metadata": {"category": "style", "niche": "all",
                     "emotion": "neutral", "performance_tier": "A"}
    })

    entries.append({
        "id": "style_emoji_001",
        "document": (
            "EMOJI RULE: Use emojis as visual anchors, not decoration. "
            "Place one emoji at the START of a key point to draw the eye: "
            "'🔬 Scientists found...' / '📊 The data shows...' / '⚡ Breaking:'. "
            "Limit to 1 emoji per line, 5-8 total per post. "
            "More than 10 emojis looks spammy and reduces credibility. "
            "Never use emojis mid-sentence — only at line starts or line ends."
        ),
        "metadata": {"category": "style", "niche": "all",
                     "emotion": "neutral", "performance_tier": "A"}
    })

    entries.append({
        "id": "style_authority_001",
        "document": (
            "AUTHORITY VOICE RULE: Write with confidence and precision. "
            "Avoid all hedging language: remove 'might', 'could', 'seems like', 'possibly', 'maybe'. "
            "Instead use: 'Scientists found', 'The data shows', 'This means', 'Here is what happened'. "
            "Confident writing builds trust faster. "
            "Hedging makes the account sound uncertain, which loses followers. "
            "Be direct. State facts. Then give your clear interpretation."
        ),
        "metadata": {"category": "style", "niche": "all",
                     "emotion": "neutral", "performance_tier": "A"}
    })

    entries.append({
        "id": "style_mobile_001",
        "document": (
            "MOBILE-FIRST RULE: 75% of Instagram users are on mobile. "
            "Only the first 3 lines are visible before the 'more' button. "
            "Those 3 lines MUST: 1) Hook with intrigue or a bold fact. "
            "2) Hint at the value inside. 3) Create a need to tap 'more'. "
            "If the first 3 lines are not compelling, the rest never gets read. "
            "Write the first 3 lines last, after drafting the full post."
        ),
        "metadata": {"category": "style", "niche": "all",
                     "emotion": "neutral", "performance_tier": "A"}
    })

    entries.append({
        "id": "style_length_001",
        "document": (
            "CAPTION LENGTH GUIDE: "
            "News and information content: 150-250 words is the sweet spot. "
            "Long enough to deliver real value, short enough to keep attention. "
            "Motivation and quotes: 80-150 words. "
            "How-to and tutorials: 250-400 words. "
            "General rule: if you can cut a sentence without losing meaning, cut it. "
            "Every word must earn its place. Tight writing outperforms padded writing."
        ),
        "metadata": {"category": "style", "niche": "all",
                     "emotion": "neutral", "performance_tier": "A"}
    })

    entries.append({
        "id": "style_specificity_001",
        "document": (
            "SPECIFICITY RULE: Specific numbers and details build credibility. "
            "BAD: 'Millions of people are affected by this.' "
            "GOOD: '34 million people worldwide are now affected — up from 12 million in 2020.' "
            "BAD: 'This happened recently.' "
            "GOOD: 'This was announced this week and takes effect in 30 days.' "
            "Specific details make content feel researched and trustworthy. "
            "Round numbers feel estimated. Precise numbers feel factual."
        ),
        "metadata": {"category": "style", "niche": "all",
                     "emotion": "neutral", "performance_tier": "A"}
    })

    # ── VIRAL FORMAT PATTERNS ──────────────────────────────────────────────────

    entries.append({
        "id": "format_listicle_001",
        "document": (
            "LISTICLE FORMAT: 'X things you did not know about [topic]' consistently outperforms. "
            "Use odd numbers: 3, 5, 7. Odd numbers feel less manufactured than even. "
            "Format each point: bold number + period + fact on its own line. "
            "Example: '1. [Fact]\\n2. [Fact]\\n3. [Fact]' "
            "Creates visual rhythm that keeps readers scrolling through all points. "
            "Works for news, science, finance, fitness, and motivation."
        ),
        "metadata": {"category": "format", "niche": "all",
                     "emotion": "curiosity", "performance_tier": "A"}
    })

    entries.append({
        "id": "format_contrarian_001",
        "document": (
            "CONTRARIAN FORMAT: Challenge what everyone believes. "
            "Structure: 'Everyone thinks [X]. But [Y] shows otherwise. Here is the truth:' "
            "Then deliver the counter-intuitive reality with evidence. "
            "End with: 'Most people will never know this. Now you do.' "
            "This format consistently drives 3x more comments than standard formats "
            "because readers either validate or push back — both increase reach."
        ),
        "metadata": {"category": "format", "niche": "all",
                     "emotion": "shock", "performance_tier": "A"}
    })

    entries.append({
        "id": "format_nobody_talking_001",
        "document": (
            "EXCLUSIVE INSIGHT FORMAT: 'Nobody is talking about this but [topic] is happening right now.' "
            "Or: 'The story behind [topic] that the media is not covering properly.' "
            "Creates an in-group feeling — readers feel they are getting exclusive information. "
            "Follow with: real facts, actual context, honest analysis. "
            "The exclusivity hook only works if the content genuinely delivers something most people missed."
        ),
        "metadata": {"category": "format", "niche": "latest_news",
                     "emotion": "curiosity", "performance_tier": "A"}
    })

    entries.append({
        "id": "format_stat_opener_001",
        "document": (
            "SHOCKING STAT OPENER FORMAT: Lead with a statistic that reframes everything. "
            "Structure: '[Specific number] [shocking fact about topic].' "
            "Then explain what it means. Then why it matters. Then CTA. "
            "Numbers stop the scroll. The brain processes specificity as credibility. "
            "Get the statistic right — false stats destroy account credibility permanently."
        ),
        "metadata": {"category": "format", "niche": "all",
                     "emotion": "shock", "performance_tier": "A"}
    })

    entries.append({
        "id": "format_before_after_001",
        "document": (
            "BEFORE/AFTER FORMAT: Show transformation or change clearly. "
            "Structure: 'Before [discovery/event]: we thought [old belief]. "
            "After [discovery/event]: we now know [new truth].' "
            "Or: 'The old way: [X]. The new way: [Y]. Here is why it matters.' "
            "Creates satisfying resolution that readers share. "
            "Works for science discoveries, technology breakthroughs, and mindset content."
        ),
        "metadata": {"category": "format", "niche": "all",
                     "emotion": "inspiration", "performance_tier": "A"}
    })

    entries.append({
        "id": "format_thread_001",
        "document": (
            "MINI-THREAD FORMAT: When covering complex topics, structure as a short thread within one post. "
            "Use numbered points: '1/', '2/', '3/' style. "
            "Each point is self-contained and reveals something new. "
            "End the last point with the most surprising or impactful fact. "
            "This format signals depth and thoroughness, earning saves and follows."
        ),
        "metadata": {"category": "format", "niche": "latest_news",
                     "emotion": "curiosity", "performance_tier": "B"}
    })

    # ── ALGORITHM INTELLIGENCE ─────────────────────────────────────────────────

    entries.append({
        "id": "algo_timing_001",
        "document": (
            "POSTING TIME INTELLIGENCE: First 60 minutes after posting are critical. "
            "The Instagram algorithm judges content quality by early engagement rate. "
            "Best universal times: Tuesday-Friday 7-9am, 12-2pm, 6-9pm local time. "
            "Avoid posting between 11pm and 6am — engagement is 40% lower. "
            "For breaking news: post immediately regardless of time — recency beats timing."
        ),
        "metadata": {"category": "style", "niche": "all",
                     "emotion": "neutral", "performance_tier": "A"}
    })

    entries.append({
        "id": "algo_consistency_001",
        "document": (
            "CONSISTENCY RULE: Posting consistently is more powerful than posting perfectly. "
            "3-5 posts per week outperforms 1 perfect post per week in follower growth. "
            "The algorithm rewards accounts that post on a regular schedule. "
            "Once you establish a posting rhythm, the algorithm starts surfacing your content "
            "to followers before they even search. Consistency compounds over time."
        ),
        "metadata": {"category": "style", "niche": "all",
                     "emotion": "neutral", "performance_tier": "A"}
    })

    entries.append({
        "id": "algo_first3lines_001",
        "document": (
            "FIRST 3 LINES FORMULA for maximum tap-through rate: "
            "Line 1: The hook — one bold, surprising, or curiosity-triggering statement. "
            "Line 2: The amplifier — deepen the intrigue or state the payoff. "
            "Line 3: The invite — a phrase that makes them tap more, like 'Here is the full story:' "
            "or 'This is bigger than it sounds. Here is why:' "
            "These 3 lines determine whether 80% of readers ever see the rest of your post."
        ),
        "metadata": {"category": "style", "niche": "all",
                     "emotion": "neutral", "performance_tier": "A"}
    })

    # ── INSERT ALL ENTRIES ─────────────────────────────────────────────────────
    ids       = [e["id"]       for e in entries]
    documents = [e["document"] for e in entries]
    metadatas = [e["metadata"] for e in entries]

    batch = 20
    for start in range(0, len(ids), batch):
        col.add(
            ids       = ids[start:start+batch],
            documents = documents[start:start+batch],
            metadatas = metadatas[start:start+batch],
        )

    print(f"  [RAG] Expert knowledge seeded: {len(ids)} entries across "
          f"hooks, frameworks, triggers, hashtags, CTAs, style tips, formats.")
