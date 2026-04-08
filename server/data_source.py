"""
SQLite-backed local datasource utilities for the Social Media Optimizer env.

The local database is seeded with synthetic data so the environment remains safe
for sharing and reproducible across machines.
"""

from __future__ import annotations

import csv
import json
import math
import random
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


PLATFORM_AUDIENCE_MAP = {
    "facebook": "consumer",
    "instagram": "youth",
    "linkedin": "b2b",
    "x": "general",
    "youtube": "consumer",
}

PLATFORM_NAME_MAP = {
    "fb": "facebook",
    "ig": "instagram",
    "linkedin": "linkedin",
    "x": "x",
    "yt": "youtube",
}

PLATFORM_ORDER = ["instagram", "facebook", "linkedin", "youtube", "x"]
SYNTHETIC_POSTS_PER_BRAND = 24
SYNTHETIC_PARENT_BRANDS = [
    ("synthetic_brand_1", "Synthetic Brand 1"),
    ("synthetic_brand_2", "Synthetic Brand 2"),
    ("synthetic_brand_3", "Synthetic Brand 3"),
    ("synthetic_brand_4", "Synthetic Brand 4"),
    ("synthetic_brand_5", "Synthetic Brand 5"),
]


def ensure_sqlite_seeded(db_path: Path, data_root: Optional[Path] = None) -> Path:
    """
    Ensure a SQLite database exists and is populated with synthetic seed data.

    If the DB already exists and contains channel profiles, it is reused.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        _create_schema(conn)
        channel_count = conn.execute("SELECT COUNT(*) FROM channel_profiles").fetchone()[0]
        if channel_count == 0:
            _seed_synthetic(conn)
    return db_path


def load_brand_channels_from_sqlite(db_path: Path, n_channels: int) -> List[Dict[str, Any]]:
    """Load channel-calibrated brand entities from SQLite."""
    db_path = Path(db_path)
    if not db_path.exists():
        return []

    query = """
        SELECT
            cp.id,
            cp.brand_slug,
            cp.brand_name,
            cp.platform,
            cp.data_source,
            cp.audience_type,
            cp.follower_count,
            cp.brand_quality,
            cp.historical_posts,
            cp.historical_average_engagement,
            cp.historical_content_mix_json
        FROM channel_profiles cp
        ORDER BY
            CASE cp.platform
                WHEN 'instagram' THEN 0
                WHEN 'facebook' THEN 1
                WHEN 'linkedin' THEN 2
                WHEN 'youtube' THEN 3
                WHEN 'x' THEN 4
                ELSE 99
            END,
            cp.historical_posts DESC,
            cp.brand_name ASC
        LIMIT ?
    """

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(query, (n_channels,)).fetchall()

    channels = []
    for row in rows:
        content_mix = json.loads(row["historical_content_mix_json"] or "{}")
        channels.append(
            {
                "brand_name": row["brand_name"],
                "platform": row["platform"],
                "brand_slug": row["brand_slug"],
                "data_source": row["data_source"],
                "audience_type": row["audience_type"],
                "follower_count": row["follower_count"],
                "brand_quality": row["brand_quality"],
                "historical_posts": row["historical_posts"],
                "historical_average_engagement": row["historical_average_engagement"],
                "historical_content_mix": content_mix,
                "engagement_history": [],
                "recent_content_types": [],
                "recent_time_slots": [],
                "content_mix": {"reel": 0, "carousel": 0, "static": 0},
                "total_posts": 0,
                "last_post_step": None,
                "last_budget_fraction": 0.0,
                "cumulative_budget_spent": 0.0,
                "budget_spent": 0.0,
                "budget_remaining": 0.0,
            }
        )
    return channels


def load_brand_channels_from_local_data(data_root: Path, n_channels: int) -> List[Dict[str, Any]]:
    """Load channel-calibrated brand entities directly from local brand CSV exports."""
    brand_root = Path(data_root) / "brand"
    if not brand_root.exists():
        return []

    profiles: List[Dict[str, Any]] = []
    for brand_dir in sorted(path for path in brand_root.iterdir() if path.is_dir()):
        for csv_path in sorted(brand_dir.glob("*.csv")):
            platform_key = PLATFORM_NAME_MAP.get(csv_path.stem.lower(), csv_path.stem.lower())
            try:
                rows = _read_csv_rows(csv_path)
            except OSError:
                continue

            summary = _summarize_channel_rows(brand_dir.name, platform_key, rows)
            if not summary:
                continue

            content_mix = json.loads(summary["historical_content_mix_json"] or "{}")
            profiles.append(
                {
                    "brand_name": summary["brand_name"],
                    "platform": summary["platform"],
                    "brand_slug": summary["brand_slug"],
                    "data_source": "local_csv",
                    "audience_type": summary["audience_type"],
                    "follower_count": summary["follower_count"],
                    "brand_quality": summary["brand_quality"],
                    "historical_posts": summary["historical_posts"],
                    "historical_average_engagement": summary["historical_average_engagement"],
                    "historical_content_mix": content_mix,
                    "engagement_history": [],
                    "recent_content_types": [],
                    "recent_time_slots": [],
                    "content_mix": {"reel": 0, "carousel": 0, "static": 0},
                    "total_posts": 0,
                    "last_post_step": None,
                    "last_budget_fraction": 0.0,
                    "cumulative_budget_spent": 0.0,
                    "budget_spent": 0.0,
                    "budget_remaining": 0.0,
                }
            )

    profiles.sort(
        key=lambda item: (
            PLATFORM_ORDER.index(item["platform"])
            if item["platform"] in PLATFORM_ORDER
            else 99,
            -int(item["historical_posts"]),
            item["brand_name"],
        )
    )
    return profiles[:n_channels]


def summarize_sqlite(db_path: Path) -> Dict[str, Any]:
    """Return a small summary of the SQLite store for debugging/tests."""
    if not Path(db_path).exists():
        return {"exists": False}

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        profile_count = conn.execute("SELECT COUNT(*) FROM channel_profiles").fetchone()[0]
        raw_post_count = conn.execute("SELECT COUNT(*) FROM raw_posts").fetchone()[0]
        platforms = [
            row["platform"]
            for row in conn.execute(
                "SELECT DISTINCT platform FROM channel_profiles ORDER BY platform"
            ).fetchall()
        ]

    return {
        "exists": True,
        "channel_profiles": profile_count,
        "raw_posts": raw_post_count,
        "platforms": platforms,
    }


def _create_schema(conn: sqlite3.Connection) -> None:
    """Create the local SQLite schema."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS raw_posts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            brand_slug TEXT NOT NULL,
            platform TEXT NOT NULL,
            source_file TEXT NOT NULL,
            post_date TEXT,
            permalink TEXT,
            title TEXT,
            description TEXT,
            native_content_type TEXT,
            content_type TEXT NOT NULL,
            reach REAL,
            impressions REAL,
            views REAL,
            likes REAL,
            comments REAL,
            shares REAL,
            saves REAL,
            clicks REAL,
            reposts REAL,
            engagement REAL,
            raw_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS channel_profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            brand_slug TEXT NOT NULL,
            brand_name TEXT NOT NULL,
            platform TEXT NOT NULL,
            data_source TEXT NOT NULL DEFAULT 'sqlite',
            audience_type TEXT NOT NULL,
            follower_count INTEGER NOT NULL,
            brand_quality REAL NOT NULL,
            historical_posts INTEGER NOT NULL,
            historical_average_engagement REAL NOT NULL,
            historical_content_mix_json TEXT NOT NULL,
            UNIQUE(brand_slug, platform)
        );
        """
    )
    conn.commit()


def _seed_from_csv(conn: sqlite3.Connection, data_root: Path) -> None:
    """Import CSV seed files into SQLite and derive channel profiles."""
    brand_root = Path(data_root) / "brand"
    if not brand_root.exists():
        return

    raw_rows: List[Dict[str, Any]] = []
    profiles: List[Dict[str, Any]] = []

    for brand_dir in sorted(path for path in brand_root.iterdir() if path.is_dir()):
        for csv_path in sorted(brand_dir.glob("*.csv")):
            platform_key = PLATFORM_NAME_MAP.get(csv_path.stem.lower(), csv_path.stem.lower())
            rows = _read_csv_rows(csv_path)
            if not rows:
                continue

            raw_rows.extend(_normalize_raw_posts(brand_dir.name, platform_key, csv_path, rows))
            summary = _summarize_channel_rows(brand_dir.name, platform_key, rows)
            if summary:
                profiles.append(summary)

    if raw_rows:
        conn.executemany(
            """
            INSERT INTO raw_posts (
                brand_slug, platform, source_file, post_date, permalink, title, description,
                native_content_type, content_type, reach, impressions, views, likes, comments,
                shares, saves, clicks, reposts, engagement, raw_json
            ) VALUES (
                :brand_slug, :platform, :source_file, :post_date, :permalink, :title, :description,
                :native_content_type, :content_type, :reach, :impressions, :views, :likes, :comments,
                :shares, :saves, :clicks, :reposts, :engagement, :raw_json
            )
            """,
            raw_rows,
        )

    if profiles:
        conn.executemany(
            """
            INSERT INTO channel_profiles (
                brand_slug, brand_name, platform, data_source, audience_type, follower_count,
                brand_quality, historical_posts, historical_average_engagement,
                historical_content_mix_json
            ) VALUES (
                :brand_slug, :brand_name, :platform, :data_source, :audience_type, :follower_count,
                :brand_quality, :historical_posts, :historical_average_engagement,
                :historical_content_mix_json
            )
            """,
            profiles,
        )

    conn.commit()


def _seed_synthetic(conn: sqlite3.Connection, seed: int = 42) -> None:
    """Populate the SQLite database with deterministic synthetic brand data."""
    rng = random.Random(seed)
    raw_rows: List[Dict[str, Any]] = []
    profiles: List[Dict[str, Any]] = []

    for brand_index, (brand_slug, brand_name) in enumerate(SYNTHETIC_PARENT_BRANDS):
        for platform_index, platform in enumerate(PLATFORM_ORDER):
            audience_type = PLATFORM_AUDIENCE_MAP.get(platform, "general")
            follower_count = rng.randint(15_000, 350_000)
            brand_quality = round(rng.uniform(0.72, 1.18), 2)
            content_counts = {"reel": 0, "carousel": 0, "static": 0}
            engagement_total = 0.0
            recent_content_types: List[str] = []
            recent_time_slots: List[int] = []

            for post_idx in range(SYNTHETIC_POSTS_PER_BRAND):
                if audience_type in {"consumer", "youth"}:
                    content_type = ["reel", "reel", "carousel", "static"][post_idx % 4]
                    time_slot = [4, 5, 3, 2][post_idx % 4]
                elif audience_type == "b2b":
                    content_type = ["carousel", "reel", "static", "carousel"][post_idx % 4]
                    time_slot = [1, 2, 1, 3][post_idx % 4]
                else:
                    content_type = ["reel", "carousel", "static"][post_idx % 3]
                    time_slot = [5, 2, 4][post_idx % 3]

                repeated_content_streak = 0
                for item in reversed(recent_content_types[-5:]):
                    if item != content_type:
                        break
                    repeated_content_streak += 1

                repeated_time_streak = 0
                for item in reversed(recent_time_slots[-5:]):
                    if item != time_slot:
                        break
                    repeated_time_streak += 1

                engagement = round(
                    max(
                        0.0,
                        min(
                            1.0,
                            0.12
                            + (0.06 * brand_quality)
                            + (0.18 if content_type == "reel" else 0.11 if content_type == "carousel" else 0.08)
                            + (0.20 if time_slot in {4, 5} and audience_type in {"consumer", "youth"} else 0.14 if time_slot in {1, 2} and audience_type == "b2b" else 0.10)
                            + (0.03 * min(post_idx, 5))
                            - (0.02 * repeated_content_streak)
                            - (0.015 * repeated_time_streak),
                        ),
                    ),
                    4,
                )

                content_counts[content_type] += 1
                engagement_total += engagement
                recent_content_types.append(content_type)
                recent_time_slots.append(time_slot)
                raw_rows.append(
                    {
                        "brand_slug": brand_slug,
                        "platform": platform,
                        "source_file": f"synthetic://{brand_slug}/{platform}.csv",
                        "post_date": f"2026-04-{(post_idx % 28) + 1:02d}",
                        "permalink": f"https://example.com/{brand_slug}/{platform}/{post_idx + 1}",
                        "title": f"Synthetic {brand_name} {platform.title()} Post {post_idx + 1}",
                        "description": f"Synthetic post {post_idx + 1} for {brand_name} on {platform}.",
                        "native_content_type": content_type,
                        "content_type": content_type,
                        "reach": round(follower_count * (0.03 + engagement * 0.20), 2),
                        "impressions": round(follower_count * (0.05 + engagement * 0.30), 2),
                        "views": round(follower_count * (0.04 + engagement * 0.25), 2),
                        "likes": round(follower_count * (0.01 + engagement * 0.08), 2),
                        "comments": round(follower_count * (0.002 + engagement * 0.015), 2),
                        "shares": round(follower_count * (0.001 + engagement * 0.01), 2),
                        "saves": round(follower_count * (0.001 + engagement * 0.008), 2),
                        "clicks": round(follower_count * (0.002 + engagement * 0.012), 2),
                        "reposts": round(follower_count * (0.0005 + engagement * 0.004), 2),
                        "engagement": engagement,
                        "raw_json": json.dumps(
                            {
                                "brand_slug": brand_slug,
                                "platform": platform,
                                "post_index": post_idx + 1,
                                "content_type": content_type,
                                "time_slot": time_slot,
                                "synthetic": True,
                            },
                            ensure_ascii=True,
                        ),
                    }
                )

            avg_engagement = engagement_total / SYNTHETIC_POSTS_PER_BRAND
            profiles.append(
                {
                    "brand_slug": brand_slug,
                    "brand_name": f"{brand_name} / {platform.title()}",
                    "platform": platform,
                    "data_source": "synthetic",
                    "audience_type": audience_type,
                    "follower_count": follower_count,
                    "brand_quality": brand_quality,
                    "historical_posts": SYNTHETIC_POSTS_PER_BRAND,
                    "historical_average_engagement": round(avg_engagement, 4),
                    "historical_content_mix_json": json.dumps(
                        content_counts, ensure_ascii=True, sort_keys=True
                    ),
                }
            )

    if raw_rows:
        conn.executemany(
            """
            INSERT INTO raw_posts (
                brand_slug, platform, source_file, post_date, permalink, title, description,
                native_content_type, content_type, reach, impressions, views, likes, comments,
                shares, saves, clicks, reposts, engagement, raw_json
            ) VALUES (
                :brand_slug, :platform, :source_file, :post_date, :permalink, :title, :description,
                :native_content_type, :content_type, :reach, :impressions, :views, :likes, :comments,
                :shares, :saves, :clicks, :reposts, :engagement, :raw_json
            )
            """,
            raw_rows,
        )

    if profiles:
        conn.executemany(
            """
            INSERT INTO channel_profiles (
                brand_slug, brand_name, platform, data_source, audience_type, follower_count,
                brand_quality, historical_posts, historical_average_engagement,
                historical_content_mix_json
            ) VALUES (
                :brand_slug, :brand_name, :platform, :data_source, :audience_type, :follower_count,
                :brand_quality, :historical_posts, :historical_average_engagement,
                :historical_content_mix_json
            )
            """,
            profiles,
        )

    conn.commit()


def _read_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _normalize_raw_posts(
    brand_slug: str,
    platform: str,
    csv_path: Path,
    rows: Iterable[Dict[str, str]],
) -> List[Dict[str, Any]]:
    """Normalize platform-specific CSV rows into the raw_posts table shape."""
    normalized_rows: List[Dict[str, Any]] = []
    for row in rows:
        headers = row.keys()
        post_date_col = _find_column(headers, ["date", "created date"])
        permalink_col = _find_column(headers, ["permalink", "post link", "link", "video link"])
        title_col = _find_column(headers, ["post title", "video name"])
        desc_col = _find_column(headers, ["description", "post", "caption"])
        type_col = _find_column(headers, ["post type", "content type", "type"])

        normalized_rows.append(
            {
                "brand_slug": brand_slug,
                "platform": platform,
                "source_file": str(csv_path),
                "post_date": row.get(post_date_col) if post_date_col else None,
                "permalink": row.get(permalink_col) if permalink_col else None,
                "title": row.get(title_col) if title_col else None,
                "description": row.get(desc_col) if desc_col else None,
                "native_content_type": row.get(type_col) if type_col else None,
                "content_type": _map_content_type(platform, (row.get(type_col) or "").strip()),
                "reach": _metric_value(row, ["reach"]),
                "impressions": _metric_value(row, ["impressions", "impression", " views ", "views"]),
                "views": _metric_value(row, ["views", " views "]),
                "likes": _metric_value(row, ["likes", "reactions"]),
                "comments": _metric_value(row, ["comments"]),
                "shares": _metric_value(row, ["shares"]),
                "saves": _metric_value(row, ["saves"]),
                "clicks": _metric_value(row, ["clicks"]),
                "reposts": _metric_value(row, ["reposts", "retweet"]),
                "engagement": _metric_value(row, ["engagement"]),
                "raw_json": json.dumps(row, ensure_ascii=True),
            }
        )
    return normalized_rows


def _summarize_channel_rows(
    brand_slug: str,
    platform: str,
    rows: List[Dict[str, str]],
) -> Optional[Dict[str, Any]]:
    """Derive one channel profile row from raw post rows."""
    if not rows:
        return None

    audience_type = PLATFORM_AUDIENCE_MAP.get(platform, "general")
    headers = rows[0].keys()
    engagement_col = _find_column(headers, ["engagement"])
    reach_col = _find_column(headers, ["reach", "impressions", "impression", " views ", "views"])
    type_col = _find_column(headers, ["post type", "content type", "type"])

    engagements = [_to_float(row.get(engagement_col)) for row in rows] if engagement_col else []
    engagements = [value for value in engagements if value is not None]

    reach_like = [_to_float(row.get(reach_col)) for row in rows] if reach_col else []
    reach_like = [value for value in reach_like if value is not None]

    content_counts = {"reel": 0, "carousel": 0, "static": 0}
    for row in rows:
        content_type = _map_content_type(platform, (row.get(type_col) or "").strip())
        content_counts[content_type] += 1

    avg_engagement = sum(engagements) / len(engagements) if engagements else 0.0
    avg_reach = sum(reach_like) / len(reach_like) if reach_like else 0.0
    normalized_engagement = avg_engagement / max(avg_reach, 1.0)
    follower_proxy = max(5000, int(math.sqrt(max(avg_reach, 1.0)) * 320))
    quality = 0.75 + min(0.55, normalized_engagement * 12.0 + math.log1p(avg_engagement) * 0.03)

    return {
        "brand_slug": brand_slug,
        "brand_name": f"{brand_slug.title()} / {platform.title()}",
        "platform": platform,
        "data_source": "sqlite",
        "audience_type": audience_type,
        "follower_count": follower_proxy,
        "brand_quality": round(quality, 2),
        "historical_posts": len(rows),
        "historical_average_engagement": round(avg_engagement, 4),
        "historical_content_mix_json": json.dumps(content_counts, ensure_ascii=True, sort_keys=True),
    }


def _metric_value(row: Dict[str, str], candidates: List[str]) -> Optional[float]:
    column = _find_column(row.keys(), candidates)
    if not column:
        return None
    return _to_float(row.get(column))


def _find_column(headers: Iterable[str], candidates: List[str]) -> Optional[str]:
    """Find the first header matching one of the lowercase candidate names."""
    normalized = {str(header).strip().lower(): header for header in headers}
    for candidate in candidates:
        key = candidate.strip().lower()
        if key in normalized:
            return normalized[key]
    return None


def _map_content_type(platform: str, raw_type: str) -> str:
    """Map platform-native content types into the env's abstract action space."""
    value = raw_type.strip().lower()
    if any(token in value for token in ["reel", "short", "video"]):
        return "reel"
    if any(token in value for token in ["article", "carousel", "long"]):
        return "carousel"
    if platform == "linkedin" and "document" in value:
        return "carousel"
    return "static"


def _to_float(value: Optional[str]) -> Optional[float]:
    """Convert a CSV numeric string to float, tolerating blanks and commas."""
    if value is None:
        return None
    cleaned = value.replace(",", "").strip()
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None
