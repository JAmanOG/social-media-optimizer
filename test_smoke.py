#!/usr/bin/env python3
"""Quick smoke test for the Social Media Optimizer environment."""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from models import SocialAction
from server.data_source import ensure_sqlite_seeded, summarize_sqlite
from server.simulation import compute_engagement, generate_brands
from server.social_media_environment import SocialMediaOptimizerEnv


def main() -> None:
    data_root = Path(os.environ.get("SOCIAL_DATA_ROOT", Path(__file__).resolve().parents[1] / "test_data"))
    sqlite_path = Path(
        os.environ.get(
            "SOCIAL_SQLITE_PATH",
            Path(__file__).resolve().parent / "data" / "social_media.db",
        )
    )

    print("Testing simulation module...")
    brands = generate_brands(3, seed=42)
    assert len(brands) == 3
    for brand in brands:
        print(
            f"  Brand: {brand['brand_name']:15s} | "
            f"{brand['audience_type']:10s} | q={brand['brand_quality']}"
        )

    rng = random.Random(42)
    slot_scores = []
    for slot in range(6):
        engagement = compute_engagement(
            audience_type="consumer",
            content_type="reel",
            time_slot=slot,
            day_of_week=2,
            budget_fraction=0.0,
            brand_quality=1.0,
            days_since_last_post=2,
            repeated_content_streak=0,
            repeated_time_streak=0,
            prior_average_engagement=0.2,
            rng=rng,
        )
        assert 0.0 <= engagement <= 1.0
        slot_scores.append(engagement)
        print(f"  Slot {slot}: engagement = {engagement:.4f}")
    assert max(slot_scores) > min(slot_scores)

    print("\nTesting environment basics...")
    env = SocialMediaOptimizerEnv(task_id=1, seed=42)
    obs = env.reset(task_id=1, seed=42)
    assert len(obs.brands) == 1
    assert obs.current_step == 0
    print(f"  Reset OK: {len(obs.brands)} brand(s), step={obs.current_step}, max={obs.max_steps}")
    print(
        f"  Data mode: {obs.metadata.get('data_mode')} | "
        f"brand={obs.brands[0].brand_name} | platform={obs.brands[0].platform}"
    )

    obs = env.step(SocialAction(brand_id=0, content_type="reel", post_time_slot=5))
    assert obs.current_step == 1
    assert obs.reward is not None
    assert hasattr(obs, "market_trend")
    assert hasattr(obs, "total_conversions")
    print(f"  Step OK: reward={obs.reward:.4f}, done={obs.done}, step={obs.current_step}")

    print("\nTesting invalid actions consume turns...")
    obs = env.reset(task_id=1, seed=42)
    invalid = env.step(SocialAction(brand_id=99, content_type="reel", post_time_slot=5))
    assert invalid.current_step == 1
    assert invalid.last_action_error is not None
    print(f"  Invalid action penalty OK: reward={invalid.reward:.4f}, step={invalid.current_step}")

    print("\nTesting Task 2 anti-neglect behavior...")
    exploit_env = SocialMediaOptimizerEnv(task_id=2, seed=42)
    exploit_obs = exploit_env.reset(task_id=2, seed=42)
    while not exploit_obs.done:
        exploit_obs = exploit_env.step(
            SocialAction(brand_id=0, content_type="reel", post_time_slot=5)
        )

    balanced_env = SocialMediaOptimizerEnv(task_id=2, seed=42)
    balanced_obs = balanced_env.reset(task_id=2, seed=42)
    while not balanced_obs.done:
        brand_id = balanced_obs.current_step % 3
        balanced_obs = balanced_env.step(
            SocialAction(brand_id=brand_id, content_type="reel", post_time_slot=5)
        )

    assert balanced_obs.metadata["grader_score"] > exploit_obs.metadata["grader_score"]
    print(
        "  Task 2 OK:"
        f" exploit={exploit_obs.metadata['grader_score']:.4f}"
        f" balanced={balanced_obs.metadata['grader_score']:.4f}"
    )

    print("\nTesting Task 3 diversification behavior...")
    uniform_env = SocialMediaOptimizerEnv(task_id=3, seed=42)
    uniform_obs = uniform_env.reset(task_id=3, seed=42)
    while not uniform_obs.done:
        uniform_obs = uniform_env.step(
            SocialAction(
                brand_id=0,
                content_type="reel",
                post_time_slot=5,
                budget_fractions=[0.2, 0.2, 0.2, 0.2, 0.2],
            )
        )

    focused_env = SocialMediaOptimizerEnv(task_id=3, seed=42)
    focused_obs = focused_env.reset(task_id=3, seed=42)
    while not focused_obs.done:
        focused_obs = focused_env.step(
            SocialAction(
                brand_id=0,
                content_type="reel",
                post_time_slot=5,
                budget_fractions=[1.0, 0.0, 0.0, 0.0, 0.0],
            )
        )

    assert uniform_obs.metadata["grader_score"] > focused_obs.metadata["grader_score"]
    print(
        "  Task 3 OK:"
        f" uniform={uniform_obs.metadata['grader_score']:.4f}"
        f" focused={focused_obs.metadata['grader_score']:.4f}"
    )

    print("\nTesting advanced dynamics (conversions + compliance)...")
    advanced_env = SocialMediaOptimizerEnv(task_id=3, seed=7)
    advanced_obs = advanced_env.reset(task_id=3, seed=7)
    while not advanced_obs.done:
        advanced_obs = advanced_env.step(
            SocialAction(
                brand_id=advanced_obs.current_step % 5,
                content_type="reel",
                post_time_slot=5,
                budget_fractions=[0.2, 0.2, 0.2, 0.2, 0.2],
            )
        )

    assert advanced_obs.metadata["episode_summary"]["total_conversions"] >= 0.0
    assert advanced_obs.metadata["episode_summary"]["total_policy_violations"] >= 0
    print(
        "  Advanced dynamics OK:"
        f" conversions={advanced_obs.metadata['episode_summary']['total_conversions']:.4f}"
        f" policy_violations={advanced_obs.metadata['episode_summary']['total_policy_violations']}"
    )

    print("\nTesting reproducibility...")
    env = SocialMediaOptimizerEnv(task_id=1, seed=100)
    obs1 = env.reset(task_id=1, seed=100)
    reward1 = env.step(SocialAction(brand_id=0, content_type="reel", post_time_slot=5)).reward

    obs2 = env.reset(task_id=1, seed=100)
    reward2 = env.step(SocialAction(brand_id=0, content_type="reel", post_time_slot=5)).reward
    assert obs1.brands[0].brand_name == obs2.brands[0].brand_name
    assert abs(reward1 - reward2) < 1e-4
    print(f"  Reproducibility OK: run1={reward1:.4f} run2={reward2:.4f}")

    if data_root.exists():
        print("\nTesting SQLite datasource mode...")
        sqlite_mode_db = sqlite_path.with_name("social_media_sqlite_mode.db")
        if sqlite_mode_db.exists():
            sqlite_mode_db.unlink()
        os.environ["SOCIAL_DATA_SOURCE"] = "sqlite"
        ensure_sqlite_seeded(sqlite_mode_db, data_root)
        db_summary = summarize_sqlite(sqlite_mode_db)
        assert db_summary["exists"] is True
        assert db_summary["channel_profiles"] >= 5

        sqlite_env = SocialMediaOptimizerEnv(task_id=3, seed=42)
        sqlite_obs = sqlite_env.reset(task_id=3, seed=42, sqlite_path=str(sqlite_mode_db))
        platforms = [brand.platform for brand in sqlite_obs.brands]
        assert sqlite_obs.metadata.get("data_mode") == "sqlite"
        assert Path(sqlite_obs.metadata.get("sqlite_path", "")).exists()
        assert "instagram" in platforms and "linkedin" in platforms
        print(
            f"  SQLite datasource OK: db={sqlite_mode_db} "
            f"profiles={db_summary['channel_profiles']} raw_posts={db_summary['raw_posts']} "
            f"platforms={platforms}"
        )

    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
