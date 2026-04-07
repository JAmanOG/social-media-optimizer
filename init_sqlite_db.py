#!/usr/bin/env python3
"""Bootstrap the local SQLite database from `test_data`."""

from __future__ import annotations

import os
from pathlib import Path

from server.data_source import ensure_sqlite_seeded, summarize_sqlite


def main() -> None:
    project_root = Path(__file__).resolve().parent
    data_root = Path(os.environ.get("SOCIAL_DATA_ROOT", project_root.parent / "test_data"))
    sqlite_path = Path(
        os.environ.get("SOCIAL_SQLITE_PATH", project_root / "data" / "social_media.db")
    )

    ensure_sqlite_seeded(sqlite_path, data_root)
    summary = summarize_sqlite(sqlite_path)

    print("SQLite database ready")
    print(f"  db_path: {sqlite_path}")
    print(f"  data_root: {data_root}")
    print(f"  channel_profiles: {summary.get('channel_profiles', 0)}")
    print(f"  raw_posts: {summary.get('raw_posts', 0)}")
    print(f"  platforms: {summary.get('platforms', [])}")


if __name__ == "__main__":
    main()
