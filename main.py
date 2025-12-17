# src/main.py
"""
ETL: Reddit (PRAW) + YouTube (YouTube Data API v3)
Writes combined_trends.csv and last_run.txt in project root.
"""

from pathlib import Path
from datetime import datetime, timezone
import os
import time
import logging
import sys
from random import uniform

import pandas as pd
import numpy as np
from dotenv import load_dotenv

# optional imports will be tested / used
try:
    import praw
except Exception:
    praw = None

try:
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
except Exception:
    build = None
    HttpError = Exception

# ---------- CONFIG ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "logs"
COMBINED_CSV = PROJECT_ROOT / "combined_trends.csv"
LAST_RUN = PROJECT_ROOT / "last_run.txt"

# tweak as needed
REDDIT_SUBREDDITS = ["technology", "Futurology", "MachineLearning"]
YT_SEARCH_QUERIES = ["AI", "ChatGPT", "technology"]
MAX_RESULTS_PER_QUERY = int(os.getenv("MAX_RESULTS_PER_QUERY", 25))  # per query
YT_API_KEY = os.getenv("YOUTUBE_API_KEY")  # will be loaded after .env

# load .env
load_dotenv(PROJECT_ROOT / ".env")
# reload YT_API_KEY after loading .env
YT_API_KEY = os.getenv("YOUTUBE_API_KEY")

# logging setup
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=str(LOG_DIR / "main.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logging.getLogger().addHandler(console)


def safe_save(df: pd.DataFrame, out_path: Path):
    tmp = out_path.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(out_path)


def detect_anomalies(df: pd.DataFrame, value_col: str = "metric", window: int = 20, z_thresh: float = 2.5):
    """Group-wise rolling z-score anomaly detection by 'source'."""
    if df is None or df.empty:
        return df
    df = df.copy()
    df[value_col] = pd.to_numeric(df[value_col].fillna(0), errors="coerce").fillna(0)
    df["zscore"] = 0.0
    df["is_anomaly"] = False
    for src, g in df.groupby("source"):
        idx = g.index
        if len(g) >= 3:
            roll_mean = g[value_col].rolling(window=window, min_periods=3).mean()
            roll_std = g[value_col].rolling(window=window, min_periods=3).std().replace(0, np.nan)
            z = (g[value_col] - roll_mean) / roll_std
            df.loc[idx, "zscore"] = z.fillna(0).values
            df.loc[idx, "is_anomaly"] = df.loc[idx, "zscore"].abs() > z_thresh
        else:
            df.loc[idx, "zscore"] = 0.0
            df.loc[idx, "is_anomaly"] = False
    return df


# ---------------- REDDIT ----------------
def fetch_reddit(subreddits=REDDIT_SUBREDDITS, limit=50):
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT", "trend_analyzer")

    if praw is None:
        logging.warning("PRAW not available; skipping Reddit ingest.")
        return pd.DataFrame()

    if not (client_id and client_secret):
        logging.warning("Reddit credentials missing; skipping Reddit ingest.")
        return pd.DataFrame()

    try:
        reddit = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent, check_for_async=False)
        records = []
        for sub in subreddits:
            try:
                for post in reddit.subreddit(sub).hot(limit=limit):
                    records.append(
                        {
                            "source": "Reddit",
                            "subsource": sub,
                            "title": getattr(post, "title", "") or "",
                            "text": getattr(post, "selftext", "") or "",
                            "metric": int(getattr(post, "score", 0) or 0),
                            "comments": int(getattr(post, "num_comments", 0) or 0),
                            "timestamp": datetime.fromtimestamp(getattr(post, "created_utc", time.time()), tz=timezone.utc),
                            "url": getattr(post, "url", None),
                            "id": f"reddit_{getattr(post, 'id', '')}",
                        }
                    )
            except Exception as e:
                logging.exception("Failed to fetch subreddit %s: %s", sub, e)
        df = pd.DataFrame.from_records(records)
        logging.info("Fetched %d Reddit items", len(df))
        return df
    except Exception as e:
        logging.exception("Reddit fetch failed: %s", e)
        return pd.DataFrame()


# ---------------- YOUTUBE API ----------------
def fetch_youtube_with_api(api_key=None, queries=None, max_results_per_query=MAX_RESULTS_PER_QUERY):
    """
    Uses YouTube Data API v3 to search queries and fetch video statistics.
    Returns DataFrame with pipeline schema.
    """
    if api_key is None:
        api_key = YT_API_KEY

    if build is None:
        logging.warning("googleapiclient not installed; skipping YouTube ingest.")
        return pd.DataFrame()

    if not api_key:
        logging.warning("YOUTUBE_API_KEY not set in .env; skipping YouTube ingest.")
        return pd.DataFrame()

    if queries is None:
        queries = YT_SEARCH_QUERIES

    try:
        youtube = build("youtube", "v3", developerKey=api_key, cache_discovery=False)
    except Exception as e:
        logging.exception("Failed to build YouTube client: %s", e)
        return pd.DataFrame()

    records = []
    for q in queries:
        try:
            collected_ids = []
            next_token = None
            # gather ids up to max_results_per_query
            while len(collected_ids) < max_results_per_query:
                req = youtube.search().list(
                    q=q,
                    part="id",
                    type="video",
                    maxResults=min(50, max_results_per_query - len(collected_ids)),
                    pageToken=next_token,
                )
                res = req.execute()
                for it in res.get("items", []):
                    vid = it.get("id", {}).get("videoId")
                    if vid:
                        collected_ids.append(vid)
                next_token = res.get("nextPageToken")
                if not next_token:
                    break
                time.sleep(0.1)  # small pause between pages
            if not collected_ids:
                logging.info("YouTube API query '%s' returned 0 video IDs", q)
                continue

            # fetch details in batches (videos.list supports up to 50 ids)
            for i in range(0, len(collected_ids), 50):
                batch = collected_ids[i : i + 50]
                try:
                    vreq = youtube.videos().list(part="snippet,statistics", id=",".join(batch), maxResults=50)
                    vres = vreq.execute()
                except HttpError as he:
                    logging.exception("YouTube videos().list HttpError for batch: %s", he)
                    # continue with next batch
                    continue

                for vi in vres.get("items", []):
                    vid = vi.get("id")
                    snip = vi.get("snippet", {}) or {}
                    stats = vi.get("statistics", {}) or {}
                    title = snip.get("title", "") or ""
                    desc = snip.get("description", "") or ""
                    pub = snip.get("publishedAt", None)
                    try:
                        # convert publishedAt to timezone-aware UTC datetime
                        timestamp = pd.to_datetime(pub)
                        if timestamp.tzinfo is None:
                            timestamp = timestamp.tz_localize("UTC")
                        else:
                            timestamp = timestamp.tz_convert("UTC")
                    except Exception:
                        timestamp = pd.Timestamp.now(tz="UTC")
                    view_count = int(stats.get("viewCount", 0)) if stats.get("viewCount") is not None else 0
                    comment_count = int(stats.get("commentCount", 0)) if stats.get("commentCount") is not None else None
                    records.append(
                        {
                            "source": "YouTube",
                            "subsource": q,
                            "title": title,
                            "text": desc,
                            "metric": view_count,
                            "comments": comment_count,
                            "timestamp": timestamp,
                            "url": f"https://youtu.be/{vid}",
                            "id": f"yt_{vid}",
                        }
                    )
            logging.info("YouTube API query '%s' yielded %d videos", q, len(collected_ids))
            # small jitter between queries to avoid quick repeated fingerprinting
            time.sleep(uniform(0.2, 1.0))
        except HttpError as he:
            logging.exception("YouTube API HttpError for query '%s': %s", q, he)
            continue
        except Exception as e:
            logging.exception("YouTube API query '%s' failed: %s", q, e)
            continue

    df = pd.DataFrame.from_records(records)
    logging.info("Total YouTube API rows: %d", len(df))
    return df


# ---------------- MAIN ----------------
def main():
    start = datetime.now(timezone.utc)
    logging.info("ETL started at %s", start.isoformat())

    # fetch sources
    reddit_df = fetch_reddit(limit=50)
    yt_df = fetch_youtube_with_api(api_key=YT_API_KEY, queries=YT_SEARCH_QUERIES, max_results_per_query=MAX_RESULTS_PER_QUERY)

    # standardize and combine
    dfs = []
    for df in (reddit_df, yt_df):
        if df is None or df.empty:
            continue
        for c in ["source", "subsource", "title", "text", "metric", "comments", "timestamp", "url", "id"]:
            if c not in df.columns:
                df[c] = None
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").fillna(pd.Timestamp.now(tz=timezone.utc))
        dfs.append(df[["source", "subsource", "title", "text", "metric", "comments", "timestamp", "url", "id"]])

    if not dfs:
        logging.warning("No data fetched from any source — writing fallback sample")
        sample = pd.DataFrame(
            [
                {
                    "source": "Sample",
                    "subsource": "sample",
                    "title": "sample trend",
                    "text": "fallback row - no sources available",
                    "metric": 1,
                    "comments": 0,
                    "timestamp": pd.Timestamp.now(tz=timezone.utc),
                    "url": None,
                    "id": "sample_1",
                }
            ]
        )
        out = detect_anomalies(sample, value_col="metric")
        out["ingested_at"] = datetime.now(timezone.utc)
        safe_save(out, COMBINED_CSV)
        LAST_RUN.write_text(datetime.now(timezone.utc).isoformat())
        logging.info("Wrote fallback sample to %s", COMBINED_CSV)
        return

    combined = pd.concat(dfs, ignore_index=True, sort=False)
    combined["metric"] = pd.to_numeric(combined["metric"].fillna(0), errors="coerce").fillna(0)
    combined["score"] = combined["metric"]  # placeholder for future weighting

    combined = detect_anomalies(combined, value_col="score", window=20, z_thresh=2.5)
    combined["ingested_at"] = datetime.now(timezone.utc)

    try:
        safe_save(combined, COMBINED_CSV)
        LAST_RUN.write_text(datetime.now(timezone.utc).isoformat())
        logging.info("Saved combined CSV (%d rows) to %s", len(combined), COMBINED_CSV)
    except Exception as e:
        logging.exception("Failed to save combined CSV: %s", e)
        raise

    end = datetime.now(timezone.utc)
    logging.info("ETL completed at %s (elapsed %.2fs)", end.isoformat(), (end - start).total_seconds())


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception("ETL failed unexpectedly: %s", e)
        sys.exit(1)
