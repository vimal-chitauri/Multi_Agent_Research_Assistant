import os
import re
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()

_CACHE: dict = {}
_CACHE_TTL   = 1800


def _cache_get(key: str):
    entry = _CACHE.get(key)
    if entry and time.time() - entry[0] < _CACHE_TTL:
        return entry[1]
    return None


def _cache_set(key: str, data):
    _CACHE[key] = (time.time(), data)


NICHE_SUBREDDITS = {
    "technology":   ["technology", "tech", "programming", "artificial", "gadgets"],
    "ai":           ["artificial", "MachineLearning", "ChatGPT", "singularity", "technology"],
    "fitness":      ["fitness", "loseit", "bodybuilding", "running", "nutrition"],
    "motivation":   ["GetMotivated", "selfimprovement", "productivity", "LifeAdvice"],
    "finance":      ["personalfinance", "investing", "wallstreetbets", "financialindependence"],
    "crypto":       ["CryptoCurrency", "Bitcoin", "ethereum", "CryptoMarkets", "defi"],
    "marketing":    ["marketing", "digital_marketing", "socialmedia", "entrepreneur", "SEO"],
    "food":         ["food", "recipes", "EatCheapAndHealthy", "MealPrepSunday", "nutrition"],
    "travel":       ["travel", "solotravel", "backpacking", "digitalnomad", "TravelHacks"],
    "latest_news":  ["news", "worldnews", "UpliftingNews", "interestingasfuck", "todayilearned"],
    "gaming":       ["gaming", "pcgaming", "PS5", "XboxSeriesX", "indiegaming"],
    "sports":       ["sports", "nba", "nfl", "soccer", "MMA"],
    "entertainment":["movies", "television", "Music", "entertainment", "popculturechat"],
    "celebrity":    ["entertainment", "popculturechat", "Music", "movies", "television"],
}

_REDDIT_MIN_UPVOTES = 500


def get_google_daily_trending(region: str = "US") -> list[dict]:
    key = f"gtrends_daily_{region}"
    cached = _cache_get(key)
    if cached is not None:
        return cached

    try:
        from pytrends.request import TrendReq
        pt = TrendReq(hl="en-US", tz=360, timeout=(10, 25))

        try:
            df = pt.realtime_trending_searches(pn="US" if region == "US" else region)
            if df is not None and not df.empty:
                topics = []
                for i, (_, row) in enumerate(df.head(20).iterrows()):
                    title = row.get("title") or row.get(0) or ""
                    if title and isinstance(title, str):
                        topics.append({
                            "topic":  title,
                            "score":  100 - (i * 4),
                            "source": "google_daily",
                        })
                if topics:
                    _cache_set(key, topics)
                    return topics
        except Exception:
            pass

        try:
            pn_map = {"US": "united_states", "GB": "united_kingdom",
                      "CA": "canada", "AU": "australia", "IN": "india"}
            pn  = pn_map.get(region, "united_states")
            df2 = pt.trending_searches(pn=pn)
            results = [
                {
                    "topic":  str(row[0]),
                    "score":  100 - (i * 4),
                    "source": "google_daily",
                }
                for i, (_, row) in enumerate(df2.iterrows())
            ][:20]
            if results:
                _cache_set(key, results)
                return results
        except Exception:
            pass

        return []
    except Exception as e:
        print(f"  [Trends] Google daily failed: {e}")
        return []


def get_google_niche_rising(niche: str, region: str = "US") -> list[dict]:
    key = f"gtrends_niche_{niche}_{region}"
    cached = _cache_get(key)
    if cached is not None:
        return cached

    try:
        from pytrends.request import TrendReq
        pt = TrendReq(hl="en-US", tz=360, timeout=(10, 25))
        pt.build_payload([niche], cat=0, timeframe="now 1-d", geo=region)
        related = pt.related_queries()
        rising  = related.get(niche, {}).get("rising")
        if rising is None or rising.empty:
            return []

        results = []
        for i, (_, row) in enumerate(rising.head(15).iterrows()):
            raw   = int(row["value"])
            score = min(raw, 200)
            results.append({
                "topic":  str(row["query"]),
                "score":  score,
                "source": "google_rising",
            })
        _cache_set(key, results)
        return results
    except Exception as e:
        print(f"  [Trends] Google rising failed: {e}")
        return []


def get_reddit_trending(niche: str, limit_per_sub: int = 25) -> list[dict]:
    key = f"reddit_{niche}"
    cached = _cache_get(key)
    if cached is not None:
        return cached

    client_id     = os.getenv("REDDIT_CLIENT_ID", "")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET", "")
    user_agent    = os.getenv("REDDIT_USER_AGENT", "TrendBot/1.0")

    if not client_id or client_id == "your_reddit_client_id":
        print("  [Reddit] REDDIT_CLIENT_ID not set — skipping Reddit signal")
        return []

    subreddits = NICHE_SUBREDDITS.get(niche.lower(), ["popular"])

    try:
        import praw
        reddit = praw.Reddit(
            client_id     = client_id,
            client_secret = client_secret,
            user_agent    = user_agent,
            read_only     = True,
        )
        results = []
        seen_titles = set()

        for sub_name in subreddits[:4]:
            try:
                sub = reddit.subreddit(sub_name)
                for post in sub.hot(limit=limit_per_sub):
                    if post.score < _REDDIT_MIN_UPVOTES:
                        continue
                    title_key = _topic_key(post.title)
                    if title_key in seen_titles:
                        continue
                    seen_titles.add(title_key)

                    engagement = post.score + post.num_comments * 10
                    results.append({
                        "topic":        post.title[:120],
                        "upvotes":      post.score,
                        "comments":     post.num_comments,
                        "engagement":   engagement,
                        "subreddit":    sub_name,
                        "url":          f"https://reddit.com{post.permalink}",
                        "source":       "reddit",
                    })
            except Exception as sub_e:
                print(f"  [Reddit] r/{sub_name} failed: {sub_e}")

        _cache_set(key, results)
        return results

    except Exception as e:
        print(f"  [Reddit] Failed: {e}")
        return []


_NEWS_CATEGORIES = ["general", "entertainment", "sports", "technology", "health", "science", "business"]


def get_multi_category_headlines(articles_per_category: int = 5) -> list[dict]:
    import random
    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        return [{"error": "NEWS_API_KEY not set", "source": "newsapi"}]

    chosen = random.sample(_NEWS_CATEGORIES, min(4, len(_NEWS_CATEGORIES)))
    all_articles: list[dict] = []
    seen_urls: set[str] = set()

    for cat in chosen:
        try:
            r = requests.get(
                "https://newsapi.org/v2/top-headlines",
                params={
                    "country":  "us",
                    "category": cat,
                    "pageSize": articles_per_category,
                    "apiKey":   api_key,
                },
                timeout=10,
            )
            r.raise_for_status()
            for a in r.json().get("articles", []):
                url   = a.get("url", "")
                title = a.get("title", "")
                if not url or url in seen_urls or not title or "[Removed]" in title:
                    continue
                seen_urls.add(url)
                all_articles.append({
                    "title":       title,
                    "description": a.get("description", ""),
                    "source":      a["source"]["name"],
                    "url":         url,
                    "published":   a["publishedAt"],
                    "category":    cat,
                    "origin":      "newsapi_headlines",
                })
        except Exception as e:
            print(f"  [News] [{cat}] error: {e}")

    import random as _r
    _r.shuffle(all_articles)
    return all_articles


def get_top_headlines(country: str = "us", page_size: int = 10) -> list[dict]:
    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        return [{"error": "NEWS_API_KEY not set", "source": "newsapi"}]
    try:
        r = requests.get(
            "https://newsapi.org/v2/top-headlines",
            params={"country": country, "pageSize": page_size, "apiKey": api_key},
            timeout=10,
        )
        r.raise_for_status()
        articles = r.json().get("articles", [])
        return [
            {
                "title":       a["title"],
                "description": a.get("description", ""),
                "source":      a["source"]["name"],
                "url":         a["url"],
                "published":   a["publishedAt"],
                "origin":      "newsapi_headlines",
            }
            for a in articles
            if a.get("title") and a.get("url") and "[Removed]" not in a.get("title", "")
        ]
    except Exception as e:
        return [{"error": str(e), "source": "newsapi"}]


def get_news_trending(query: str, language: str = "en") -> list[dict]:
    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        return [{"error": "NEWS_API_KEY not set", "source": "newsapi"}]
    try:
        r = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q":        query,
                "language": language,
                "sortBy":   "popularity",
                "pageSize": 10,
                "apiKey":   api_key,
            },
            timeout=10,
        )
        r.raise_for_status()
        articles = r.json().get("articles", [])
        return [
            {
                "title":       a["title"],
                "description": a.get("description", ""),
                "source":      a["source"]["name"],
                "url":         a["url"],
                "published":   a["publishedAt"],
                "origin":      "newsapi_search",
            }
            for a in articles
        ]
    except Exception as e:
        return [{"error": str(e), "source": "newsapi"}]


def get_hackernews_trending(limit: int = 10) -> list[dict]:
    try:
        top_ids = requests.get(
            "https://hacker-news.firebaseio.com/v0/topstories.json", timeout=10
        ).json()[:limit]
        posts = []
        for sid in top_ids:
            s = requests.get(
                f"https://hacker-news.firebaseio.com/v0/item/{sid}.json", timeout=10
            ).json()
            if s and s.get("title"):
                posts.append({
                    "title":    s["title"],
                    "score":    s.get("score", 0),
                    "comments": s.get("descendants", 0),
                    "url":      s.get("url", f"https://news.ycombinator.com/item?id={sid}"),
                    "source":   "hackernews",
                })
        return posts
    except Exception as e:
        return [{"error": str(e), "source": "hackernews"}]


_STOPWORDS = {"a", "an", "the", "in", "on", "at", "by", "for", "of", "to",
              "is", "was", "are", "has", "it", "its", "this", "that", "with",
              "from", "as", "and", "or", "but", "not", "so", "how", "new",
              "says", "say", "report", "reported", "after", "over", "top"}


def _topic_key(title: str) -> str:
    words = re.sub(r"[^\w\s]", "", title.lower()).split()
    content = [w for w in words if w not in _STOPWORDS and len(w) > 1]
    return " ".join(content[:3])


def _normalize(values: list[float]) -> list[float]:
    if not values:
        return values
    lo, hi = min(values), max(values)
    if hi == lo:
        return [50.0] * len(values)
    return [100.0 * (v - lo) / (hi - lo) for v in values]


def score_topics(
    google_daily:  list[dict],
    google_rising: list[dict],
    reddit_posts:  list[dict],
    news_articles: list[dict],
    hackernews:    list[dict] | None = None,
    weights:       dict | None = None,
) -> list[dict]:
    w = weights or {"google": 0.40, "reddit": 0.35, "news": 0.25}

    google_raw = {item["topic"]: item["score"] for item in (google_daily + google_rising)}

    reddit_engagements = [p["engagement"] for p in reddit_posts] or [0]
    reddit_norm_vals   = _normalize(reddit_engagements)
    reddit_norm        = {p["topic"]: s for p, s in zip(reddit_posts, reddit_norm_vals)}

    g_vals     = list(google_raw.values()) or [0]
    g_norm_raw = _normalize(g_vals)
    g_norm     = {t: s for t, s in zip(google_raw.keys(), g_norm_raw)}

    news_title_keys = [_topic_key(a["title"]) for a in news_articles]

    candidates: dict[str, dict] = {}

    def _upsert(topic_text: str, g: float = 0, r: float = 0,
                n: float = 0, source: str = "", extra: dict = None):
        key = _topic_key(topic_text)
        if key not in candidates:
            candidates[key] = {
                "topic":   topic_text,
                "g_score": g,
                "r_score": r,
                "n_score": n,
                "sources": set(),
                "extra":   extra or {},
            }
        else:
            candidates[key]["g_score"] = max(candidates[key]["g_score"], g)
            candidates[key]["r_score"] = max(candidates[key]["r_score"], r)
            candidates[key]["n_score"] = max(candidates[key]["n_score"], n)
            if extra:
                candidates[key]["extra"].update(extra)
        if source:
            candidates[key]["sources"].add(source)

    for topic, g_sc in g_norm.items():
        _upsert(topic, g=g_sc, source="google")

    for post in reddit_posts:
        r_sc  = reddit_norm.get(post["topic"], 0)
        rkey  = _topic_key(post["topic"])
        g_sc  = max((g_norm.get(t, 0) for t in g_norm if _topic_key(t) == rkey), default=0)
        _upsert(post["topic"], g=g_sc, r=r_sc, source="reddit",
                extra={"reddit_upvotes": post["upvotes"], "reddit_url": post.get("url")})

    for article in news_articles:
        if "error" in article:
            continue
        title = article["title"]
        akey  = _topic_key(title)
        n_sc  = min(100.0, news_title_keys.count(akey) * 33.0)
        _upsert(title, n=n_sc, source="news",
                extra={"news_url": article.get("url"), "news_source": article.get("source"),
                       "news_description": article.get("description", "")})

    if hackernews:
        hn_scores = [h.get("score", 0) for h in hackernews if "error" not in h] or [0]
        hn_norm_v = _normalize(hn_scores)
        for post, hn_norm in zip([h for h in hackernews if "error" not in h], hn_norm_v):
            _upsert(post["title"], n=hn_norm * 0.5, source="hackernews",
                    extra={"news_url": post.get("url"), "news_source": "Hacker News"})

    results = []
    for key, c in candidates.items():
        base_score = (
            c["g_score"] * w.get("google", 0.40)
            + c["r_score"] * w.get("reddit", 0.35)
            + c["n_score"] * w.get("news",   0.25)
        )
        n_sources   = len(c["sources"])
        cross_boost = max(0, (n_sources - 1)) * 12
        final_score = min(100.0, base_score + cross_boost)

        results.append({
            "topic":     c["topic"],
            "score":     round(final_score, 1),
            "sources":   sorted(c["sources"]),
            "n_sources": n_sources,
            "g_score":   round(c["g_score"], 1),
            "r_score":   round(c["r_score"], 1),
            "n_score":   round(c["n_score"], 1),
            **c["extra"],
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def fetch_all_signals(niche: str, parallel: bool = True) -> dict:
    t0 = time.time()

    if parallel:
        jobs = {
            "google_daily":  (get_google_daily_trending, []),
            "google_rising": (get_google_niche_rising,   [niche]),
            "reddit":        (get_reddit_trending,        [niche]),
            "news":          (get_news_trending,          [niche]),
            "hackernews":    (get_hackernews_trending,    []),
        }
        results = {}
        with ThreadPoolExecutor(max_workers=5) as ex:
            futures = {ex.submit(fn, *args): name for name, (fn, args) in jobs.items()}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    results[name] = future.result()
                except Exception as e:
                    print(f"  [Trends] {name} fetch failed: {e}")
                    results[name] = []
    else:
        results = {
            "google_daily":  get_google_daily_trending(),
            "google_rising": get_google_niche_rising(niche),
            "reddit":        get_reddit_trending(niche),
            "news":          get_news_trending(niche),
            "hackernews":    get_hackernews_trending(),
        }

    results["fetch_ms"] = round((time.time() - t0) * 1000)
    return results


def get_scored_trends(niche: str, top_n: int = 10, min_score: float = 5.0) -> list[dict]:
    signals = fetch_all_signals(niche)
    scored  = score_topics(
        google_daily  = signals.get("google_daily",  []),
        google_rising = signals.get("google_rising", []),
        reddit_posts  = signals.get("reddit",        []),
        news_articles = signals.get("news",          []),
        hackernews    = signals.get("hackernews",    []),
    )
    filtered = [t for t in scored if t["score"] >= min_score]
    return filtered[:top_n]
