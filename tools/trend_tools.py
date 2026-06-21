import os
import re
import time
import random
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
    "technology":    ["technology", "tech", "programming", "artificial", "gadgets"],
    "ai":            ["artificial", "MachineLearning", "ChatGPT", "singularity", "technology"],
    "fitness":       ["fitness", "loseit", "bodybuilding", "running", "nutrition"],
    "motivation":    ["GetMotivated", "selfimprovement", "productivity", "LifeAdvice"],
    "finance":       ["personalfinance", "investing", "wallstreetbets", "financialindependence"],
    "crypto":        ["CryptoCurrency", "Bitcoin", "ethereum", "CryptoMarkets", "defi"],
    "marketing":     ["marketing", "digital_marketing", "socialmedia", "entrepreneur", "SEO"],
    "food":          ["food", "recipes", "EatCheapAndHealthy", "MealPrepSunday", "nutrition"],
    "travel":        ["travel", "solotravel", "backpacking", "digitalnomad", "TravelHacks"],
    "latest_news":   ["news", "worldnews", "UpliftingNews", "interestingasfuck", "todayilearned"],
    "gaming":        ["gaming", "pcgaming", "PS5", "XboxSeriesX", "indiegaming"],
    "sports":        ["sports", "nba", "nfl", "soccer", "MMA"],
    "entertainment": ["movies", "television", "Music", "entertainment", "popculturechat"],
    "celebrity":     ["entertainment", "popculturechat", "Music", "movies", "television"],
}

_REDDIT_MIN_UPVOTES = 500

_YOUTUBE_CATEGORY_MAP = {
    "technology":    "28",
    "ai":            "28",
    "gaming":        "20",
    "fitness":       "17",
    "sports":        "17",
    "entertainment": "24",
    "latest_news":   "25",
    "finance":       "25",
    "crypto":        "25",
    "food":          "26",
    "travel":        "19",
    "motivation":    "22",
    "marketing":     "28",
    "music":         "10",
}

_TWITTER_NICHE_QUERIES = {
    "technology":    "(tech OR software OR AI OR gadgets) lang:en",
    "ai":            "(\"artificial intelligence\" OR ChatGPT OR LLM OR #AI) lang:en",
    "crypto":        "(crypto OR bitcoin OR ethereum OR #crypto) lang:en",
    "finance":       "(\"stock market\" OR investing OR #finance OR economy) lang:en",
    "fitness":       "(fitness OR workout OR gym OR #health) lang:en",
    "motivation":    "(motivation OR success OR mindset OR #motivation) lang:en",
    "marketing":     "(marketing OR branding OR \"social media\" OR #marketing) lang:en",
    "food":          "(recipe OR cooking OR foodie OR #food) lang:en",
    "travel":        "(travel OR wanderlust OR adventure OR #travel) lang:en",
    "latest_news":   "(breaking OR trending OR #news) lang:en",
    "gaming":        "(gaming OR gamer OR #gaming OR PlayStation OR Xbox) lang:en",
    "sports":        "(sports OR NBA OR NFL OR soccer OR #sports) lang:en",
    "entertainment": "(movies OR celebrity OR Hollywood OR #entertainment) lang:en",
}

_RSS_FEEDS = {
    "technology": [
        "https://techcrunch.com/feed/",
        "https://www.theverge.com/rss/index.xml",
        "https://feeds.arstechnica.com/arstechnica/index",
        "https://www.wired.com/feed/rss",
    ],
    "ai": [
        "https://techcrunch.com/category/artificial-intelligence/feed/",
        "https://www.theverge.com/ai-artificial-intelligence/rss/index.xml",
        "https://feeds.arstechnica.com/arstechnica/index",
    ],
    "finance": [
        "https://feeds.reuters.com/reuters/businessNews",
        "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml",
        "https://www.cnbc.com/id/10000664/device/rss/rss.html",
    ],
    "crypto": [
        "https://cointelegraph.com/rss",
        "https://coindesk.com/arc/outboundfeeds/rss/",
        "https://decrypt.co/feed",
    ],
    "sports": [
        "https://www.espn.com/espn/rss/news",
        "https://rss.nytimes.com/services/xml/rss/nyt/Sports.xml",
        "https://feeds.reuters.com/reuters/sportsNews",
    ],
    "entertainment": [
        "https://variety.com/feed/",
        "https://deadline.com/feed/",
        "https://rss.nytimes.com/services/xml/rss/nyt/Arts.xml",
    ],
    "gaming": [
        "https://www.ign.com/articles.rss",
        "https://kotaku.com/rss",
        "https://www.polygon.com/rss/index.xml",
    ],
    "fitness": [
        "https://www.menshealth.com/rss/all.xml/",
        "https://www.womenshealthmag.com/rss/all.xml/",
    ],
    "food": [
        "https://feeds.seriouseats.com/seriouseats/recipes",
        "https://www.foodnetwork.com/feeds/blog/recipe-of-the-day.xml",
    ],
    "travel": [
        "https://www.lonelyplanet.com/news/feed.rss",
        "https://feeds.reuters.com/reuters/travelNews",
    ],
    "latest_news": [
        "http://feeds.bbci.co.uk/news/rss.xml",
        "https://feeds.reuters.com/reuters/topNews",
        "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
        "https://feeds.washingtonpost.com/rss/world",
    ],
    "marketing": [
        "https://contentmarketinginstitute.com/feed/",
        "https://moz.com/blog/feed",
    ],
    "motivation": [
        "https://feeds.feedburner.com/tinybuddha",
        "http://feeds.feedburner.com/marcandangel",
    ],
}

_RSS_DEFAULT = [
    "http://feeds.bbci.co.uk/news/rss.xml",
    "https://feeds.reuters.com/reuters/topNews",
]

_NEWS_CATEGORIES = ["general", "entertainment", "sports", "technology", "health", "science", "business"]

_STOPWORDS = {
    "a", "an", "the", "in", "on", "at", "by", "for", "of", "to",
    "is", "was", "are", "has", "it", "its", "this", "that", "with",
    "from", "as", "and", "or", "but", "not", "so", "how", "new",
    "says", "say", "report", "reported", "after", "over", "top",
}


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


def get_google_daily_trending(region: str = "US") -> list[dict]:
    key = f"gtrends_daily_{region}"
    cached = _cache_get(key)
    if cached is not None:
        return cached

    for attempt in range(3):
        try:
            from pytrends.request import TrendReq
            pt = TrendReq(hl="en-US", tz=360, timeout=(10, 25), retries=2, backoff_factor=0.5)

            try:
                df = pt.realtime_trending_searches(pn="US" if region == "US" else region)
                if df is not None and not df.empty:
                    topics = []
                    for i, (_, row) in enumerate(df.head(20).iterrows()):
                        title = row.get("title") or row.get(0) or ""
                        if title and isinstance(title, str):
                            topics.append({"topic": title, "score": 100 - (i * 4), "source": "google_daily"})
                    if topics:
                        _cache_set(key, topics)
                        return topics
            except Exception:
                pass

            pn_map = {"US": "united_states", "GB": "united_kingdom",
                      "CA": "canada", "AU": "australia", "IN": "india"}
            pn  = pn_map.get(region, "united_states")
            df2 = pt.trending_searches(pn=pn)
            results = [
                {"topic": str(row[0]), "score": 100 - (i * 4), "source": "google_daily"}
                for i, (_, row) in enumerate(df2.iterrows())
            ][:20]
            if results:
                _cache_set(key, results)
                return results

        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt + random.uniform(0, 1))
            else:
                print(f"  [Google Trends] Daily failed after {attempt+1} attempts: {e}")

    return []


def get_google_niche_rising(niche: str, region: str = "US") -> list[dict]:
    key = f"gtrends_niche_{niche}_{region}"
    cached = _cache_get(key)
    if cached is not None:
        return cached

    for attempt in range(3):
        try:
            from pytrends.request import TrendReq
            pt = TrendReq(hl="en-US", tz=360, timeout=(10, 25), retries=2, backoff_factor=0.5)
            pt.build_payload([niche], cat=0, timeframe="now 1-d", geo=region)
            related = pt.related_queries()
            rising  = related.get(niche, {}).get("rising")
            if rising is None or rising.empty:
                return []
            results = []
            for i, (_, row) in enumerate(rising.head(15).iterrows()):
                results.append({
                    "topic":  str(row["query"]),
                    "score":  min(int(row["value"]), 200),
                    "source": "google_rising",
                })
            _cache_set(key, results)
            return results

        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt + random.uniform(0, 1))
            else:
                print(f"  [Google Trends] Rising failed after {attempt+1} attempts: {e}")

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
        print("  [Reddit] REDDIT_CLIENT_ID not set — skipping")
        return []

    subreddits = NICHE_SUBREDDITS.get(niche.lower(), ["popular"])

    try:
        import praw
        reddit  = praw.Reddit(client_id=client_id, client_secret=client_secret,
                              user_agent=user_agent, read_only=True)
        results = []
        seen    = set()

        for sub_name in subreddits[:4]:
            try:
                for post in reddit.subreddit(sub_name).hot(limit=limit_per_sub):
                    if post.score < _REDDIT_MIN_UPVOTES:
                        continue
                    tk = _topic_key(post.title)
                    if tk in seen:
                        continue
                    seen.add(tk)
                    results.append({
                        "topic":      post.title[:120],
                        "upvotes":    post.score,
                        "comments":   post.num_comments,
                        "engagement": post.score + post.num_comments * 10,
                        "subreddit":  sub_name,
                        "url":        f"https://reddit.com{post.permalink}",
                        "source":     "reddit",
                    })
            except Exception as sub_e:
                print(f"  [Reddit] r/{sub_name} failed: {sub_e}")

        _cache_set(key, results)
        return results

    except Exception as e:
        print(f"  [Reddit] Failed: {e}")
        return []


def get_youtube_trending(niche: str, region: str = "US") -> list[dict]:
    key = f"youtube_{niche}_{region}"
    cached = _cache_get(key)
    if cached is not None:
        return cached

    api_key = os.getenv("YOUTUBE_API_KEY", "")
    if not api_key:
        print("  [YouTube] YOUTUBE_API_KEY not set — skipping")
        return []

    params = {
        "part":       "snippet,statistics",
        "chart":      "mostPopular",
        "regionCode": region,
        "maxResults": 25,
        "key":        api_key,
    }
    cat_id = _YOUTUBE_CATEGORY_MAP.get(niche.lower())
    if cat_id:
        params["videoCategoryId"] = cat_id

    try:
        r = requests.get(
            "https://www.googleapis.com/youtube/v3/videos",
            params=params,
            timeout=10,
        )
        r.raise_for_status()
        results = []
        for item in r.json().get("items", []):
            snippet = item.get("snippet", {})
            stats   = item.get("statistics", {})
            title   = snippet.get("title", "")
            if not title:
                continue
            views      = int(stats.get("viewCount", 0))
            likes      = int(stats.get("likeCount", 0))
            comments   = int(stats.get("commentCount", 0))
            engagement = views // 1000 + likes * 5 + comments * 10
            results.append({
                "topic":      title,
                "score":      min(100, engagement // 500),
                "views":      views,
                "likes":      likes,
                "channel":    snippet.get("channelTitle", ""),
                "video_id":   item.get("id", ""),
                "source":     "youtube",
            })
        _cache_set(key, results)
        print(f"  [YouTube] {len(results)} trending videos fetched")
        return results

    except Exception as e:
        print(f"  [YouTube] Failed: {e}")
        return []


def get_twitter_trending(niche: str) -> list[dict]:
    key = f"twitter_{niche}"
    cached = _cache_get(key)
    if cached is not None:
        return cached

    bearer_token = os.getenv("TWITTER_BEARER_TOKEN", "")
    if not bearer_token:
        print("  [Twitter] TWITTER_BEARER_TOKEN not set — skipping")
        return []

    query = _TWITTER_NICHE_QUERIES.get(niche.lower(), f"{niche} lang:en")

    try:
        import tweepy
        client = tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=False)
        response = client.search_recent_tweets(
            query        = f"{query} -is:retweet",
            max_results  = 100,
            tweet_fields = ["public_metrics", "entities"],
        )
        if not response.data:
            return []

        hashtag_scores: dict[str, float] = {}
        topic_scores:   dict[str, float] = {}

        for tweet in response.data:
            metrics    = tweet.public_metrics or {}
            engagement = (
                metrics.get("like_count", 0) * 3
                + metrics.get("retweet_count", 0) * 5
                + metrics.get("reply_count", 0) * 2
            )
            entities = tweet.entities or {}
            for ht in (entities.get("hashtags") or []):
                tag = (ht.get("tag") or "").lower()
                if tag and len(tag) > 2:
                    hashtag_scores[tag] = hashtag_scores.get(tag, 0) + engagement
            if engagement > 50:
                tk = _topic_key(tweet.text)
                if tk:
                    topic_scores[tk] = topic_scores.get(tk, 0) + engagement

        results = []
        seen = set()
        for tag, score in sorted(hashtag_scores.items(), key=lambda x: -x[1])[:10]:
            results.append({"topic": f"#{tag}", "score": min(100, int(score // 20)), "source": "twitter"})
            seen.add(tag)
        for topic, score in sorted(topic_scores.items(), key=lambda x: -x[1])[:5]:
            if topic not in seen:
                results.append({"topic": topic, "score": min(100, int(score // 20)), "source": "twitter"})

        _cache_set(key, results)
        print(f"  [Twitter] {len(results)} trending topics fetched")
        return results

    except tweepy.errors.Forbidden:
        print("  [Twitter] Free tier does not support read access — skipping (Basic plan $100/mo required)")
        return []
    except tweepy.errors.TooManyRequests:
        print("  [Twitter] Rate limited — skipping")
        return []
    except Exception as e:
        if "402" in str(e) or "Payment" in str(e) or "credits" in str(e).lower():
            print("  [Twitter] No API credits — skipping (Basic plan required for reads)")
        else:
            print(f"  [Twitter] Failed: {e}")
        return []


def get_rss_trending(niche: str) -> list[dict]:
    key = f"rss_{niche}"
    cached = _cache_get(key)
    if cached is not None:
        return cached

    try:
        import feedparser
    except ImportError:
        print("  [RSS] feedparser not installed — run: pip install feedparser")
        return []

    feeds   = _RSS_FEEDS.get(niche.lower(), _RSS_DEFAULT)
    results = []
    seen    = set()

    for feed_url in feeds[:4]:
        try:
            feed = feedparser.parse(feed_url)
            source_name = feed.feed.get("title", "RSS")
            for entry in feed.entries[:8]:
                title = entry.get("title", "").strip()
                if not title:
                    continue
                tk = _topic_key(title)
                if tk in seen:
                    continue
                seen.add(tk)
                results.append({
                    "topic":       title,
                    "description": entry.get("summary", "")[:300],
                    "url":         entry.get("link", ""),
                    "source_name": source_name,
                    "published":   entry.get("published", ""),
                    "source":      "rss",
                })
        except Exception as e:
            print(f"  [RSS] {feed_url} failed: {e}")

    _cache_set(key, results)
    print(f"  [RSS] {len(results)} articles fetched")
    return results


def get_multi_category_headlines(articles_per_category: int = 5) -> list[dict]:
    import random as _r
    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        return [{"error": "NEWS_API_KEY not set", "source": "newsapi"}]

    chosen      = _r.sample(_NEWS_CATEGORIES, min(4, len(_NEWS_CATEGORIES)))
    all_articles: list[dict] = []
    seen_urls:    set[str]   = set()

    for cat in chosen:
        try:
            r = requests.get(
                "https://newsapi.org/v2/top-headlines",
                params={"country": "us", "category": cat,
                        "pageSize": articles_per_category, "apiKey": api_key},
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

    import random as _r2
    _r2.shuffle(all_articles)
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
        return [
            {
                "title":       a["title"],
                "description": a.get("description", ""),
                "source":      a["source"]["name"],
                "url":         a["url"],
                "published":   a["publishedAt"],
                "origin":      "newsapi_headlines",
            }
            for a in r.json().get("articles", [])
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
            params={"q": query, "language": language,
                    "sortBy": "popularity", "pageSize": 10, "apiKey": api_key},
            timeout=10,
        )
        r.raise_for_status()
        return [
            {
                "title":       a["title"],
                "description": a.get("description", ""),
                "source":      a["source"]["name"],
                "url":         a["url"],
                "published":   a["publishedAt"],
                "origin":      "newsapi_search",
            }
            for a in r.json().get("articles", [])
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


def score_topics(
    google_daily:   list[dict],
    google_rising:  list[dict],
    reddit_posts:   list[dict],
    news_articles:  list[dict],
    hackernews:     list[dict] | None = None,
    youtube_videos: list[dict] | None = None,
    twitter_topics: list[dict] | None = None,
    rss_articles:   list[dict] | None = None,
    weights:        dict | None = None,
) -> list[dict]:
    w = weights or {
        "google":   0.30,
        "reddit":   0.20,
        "news":     0.15,
        "youtube":  0.20,
        "twitter":  0.10,
        "rss":      0.05,
    }

    google_raw       = {item["topic"]: item["score"] for item in (google_daily + google_rising)}
    g_vals           = list(google_raw.values()) or [0]
    g_norm           = {t: s for t, s in zip(google_raw.keys(), _normalize(g_vals))}

    reddit_norm      = {}
    if reddit_posts:
        reddit_eng   = [p["engagement"] for p in reddit_posts] or [0]
        reddit_nv    = _normalize(reddit_eng)
        reddit_norm  = {p["topic"]: s for p, s in zip(reddit_posts, reddit_nv)}

    yt_norm          = {}
    if youtube_videos:
        yt_scores    = [v.get("score", 0) for v in youtube_videos] or [0]
        yt_nv        = _normalize(yt_scores)
        yt_norm      = {v["topic"]: s for v, s in zip(youtube_videos, yt_nv)}

    tw_norm          = {}
    if twitter_topics:
        tw_scores    = [t.get("score", 0) for t in twitter_topics] or [0]
        tw_nv        = _normalize(tw_scores)
        tw_norm      = {t["topic"]: s for t, s in zip(twitter_topics, tw_nv)}

    news_title_keys  = [_topic_key(a["title"]) for a in news_articles]
    rss_title_keys   = [_topic_key(a["topic"]) for a in (rss_articles or [])]

    candidates: dict[str, dict] = {}

    def _upsert(topic_text: str, g: float = 0, r: float = 0, n: float = 0,
                yt: float = 0, tw: float = 0, rss: float = 0,
                source: str = "", extra: dict = None):
        key = _topic_key(topic_text)
        if not key:
            return
        if key not in candidates:
            candidates[key] = {
                "topic":    topic_text,
                "g_score":  g,
                "r_score":  r,
                "n_score":  n,
                "yt_score": yt,
                "tw_score": tw,
                "rss_score":rss,
                "sources":  set(),
                "extra":    extra or {},
            }
        else:
            c = candidates[key]
            c["g_score"]   = max(c["g_score"],   g)
            c["r_score"]   = max(c["r_score"],   r)
            c["n_score"]   = max(c["n_score"],   n)
            c["yt_score"]  = max(c["yt_score"],  yt)
            c["tw_score"]  = max(c["tw_score"],  tw)
            c["rss_score"] = max(c["rss_score"], rss)
            if extra:
                c["extra"].update(extra)
        if source:
            candidates[key]["sources"].add(source)

    for topic, g_sc in g_norm.items():
        _upsert(topic, g=g_sc, source="google")

    for post in reddit_posts:
        r_sc = reddit_norm.get(post["topic"], 0)
        g_sc = max((g_norm.get(t, 0) for t in g_norm if _topic_key(t) == _topic_key(post["topic"])), default=0)
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
        hn_nv     = _normalize(hn_scores)
        for post, hn_sc in zip([h for h in hackernews if "error" not in h], hn_nv):
            _upsert(post["title"], n=hn_sc * 0.5, source="hackernews",
                    extra={"news_url": post.get("url"), "news_source": "Hacker News"})

    if youtube_videos:
        for v in youtube_videos:
            yt_sc = yt_norm.get(v["topic"], 0)
            _upsert(v["topic"], yt=yt_sc, source="youtube",
                    extra={"yt_views": v.get("views", 0), "yt_channel": v.get("channel", ""),
                           "yt_video_id": v.get("video_id", "")})

    if twitter_topics:
        for t in twitter_topics:
            tw_sc = tw_norm.get(t["topic"], 0)
            _upsert(t["topic"], tw=tw_sc, source="twitter")

    if rss_articles:
        for a in rss_articles:
            akey  = _topic_key(a["topic"])
            rss_sc = min(100.0, rss_title_keys.count(akey) * 33.0 + 20.0)
            _upsert(a["topic"], rss=rss_sc, source="rss",
                    extra={"news_url": a.get("url"), "news_source": a.get("source_name", "RSS"),
                           "news_description": a.get("description", "")})

    results = []
    for key, c in candidates.items():
        base_score = (
            c["g_score"]   * w.get("google",  0.30)
            + c["r_score"] * w.get("reddit",  0.20)
            + c["n_score"] * w.get("news",    0.15)
            + c["yt_score"]* w.get("youtube", 0.20)
            + c["tw_score"]* w.get("twitter", 0.10)
            + c["rss_score"]* w.get("rss",    0.05)
        )
        cross_boost = max(0, len(c["sources"]) - 1) * 12
        final_score = min(100.0, base_score + cross_boost)

        results.append({
            "topic":     c["topic"],
            "score":     round(final_score, 1),
            "sources":   sorted(c["sources"]),
            "n_sources": len(c["sources"]),
            "g_score":   round(c["g_score"],    1),
            "r_score":   round(c["r_score"],    1),
            "n_score":   round(c["n_score"],    1),
            "yt_score":  round(c["yt_score"],   1),
            "tw_score":  round(c["tw_score"],   1),
            "rss_score": round(c["rss_score"],  1),
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
            "youtube":       (get_youtube_trending,       [niche]),
            "twitter":       (get_twitter_trending,       [niche]),
            "rss":           (get_rss_trending,           [niche]),
        }
        results = {}
        with ThreadPoolExecutor(max_workers=8) as ex:
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
            "youtube":       get_youtube_trending(niche),
            "twitter":       get_twitter_trending(niche),
            "rss":           get_rss_trending(niche),
        }

    results["fetch_ms"] = round((time.time() - t0) * 1000)
    return results


def get_scored_trends(niche: str, top_n: int = 10, min_score: float = 5.0) -> list[dict]:
    signals = fetch_all_signals(niche)
    scored  = score_topics(
        google_daily   = signals.get("google_daily",  []),
        google_rising  = signals.get("google_rising", []),
        reddit_posts   = signals.get("reddit",        []),
        news_articles  = signals.get("news",          []),
        hackernews     = signals.get("hackernews",    []),
        youtube_videos = signals.get("youtube",       []),
        twitter_topics = signals.get("twitter",       []),
        rss_articles   = signals.get("rss",           []),
    )
    return [t for t in scored if t["score"] >= min_score][:top_n]
