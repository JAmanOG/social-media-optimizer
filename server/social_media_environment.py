"""
Social Media Optimizer — Core RL Environment.

Implements `reset()`, `step()`, and `state` following the OpenEnv spec.
The environment models three progressively harder social-media planning tasks.
"""

from __future__ import annotations

import math
import os
import random
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from openenv.core.env_server import Environment

# Support both in-repo and standalone imports
try:
    from ..models import BrandState, SocialAction, SocialObservation, SocialState
    from .data_source import ensure_sqlite_seeded, load_brand_channels_from_sqlite
    from .simulation import (
        compute_engagement,
        compute_paid_engagement_lift,
        generate_brands,
    )
except ImportError:
    from models import BrandState, SocialAction, SocialObservation, SocialState
    from server.data_source import ensure_sqlite_seeded, load_brand_channels_from_sqlite
    from server.simulation import (
        compute_engagement,
        compute_paid_engagement_lift,
        generate_brands,
    )

try:
    from openenv.core.env_server.types import EnvironmentMetadata
except ImportError:
    EnvironmentMetadata = None  # type: ignore[assignment,misc]


TASK_CONFIG = {
    1: {
        "n_brands": 1,
        "max_steps": 7,
        "daily_budget": 0.0,
        "name": "Single Brand Engagement",
        "target_posts_per_brand": 7,
    },
    2: {
        "n_brands": 3,
        "max_steps": 14,
        "daily_budget": 0.0,
        "name": "Multi-Brand Content Mix",
        "target_posts_per_brand": 4,
    },
    3: {
        "n_brands": 5,
        "max_steps": 21,
        "daily_budget": 1000.0,
        "name": "Budget Allocation + Cross-Brand",
        "target_posts_per_brand": 4,
    },
}

VALID_CONTENT_TYPES = {"reel", "carousel", "static"}
DEFAULT_SQLITE_PATH = Path(__file__).resolve().parents[1] / "data" / "social_media.db"


class SocialMediaOptimizerEnv(Environment):
    """
    RL environment for multi-brand social media agency optimization.

    Each step represents one planning day. The agent chooses which brand gets
    the featured organic post and, in Task 3, how to distribute that day's paid
    budget across the full portfolio.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def get_metadata(self):  # type: ignore[override]
        """Return human-readable metadata for this environment."""
        if EnvironmentMetadata is None:
            return super().get_metadata()
        return EnvironmentMetadata(
            name="Social Media Optimizer",
            description=(
                "Multi-brand social media agency RL environment. "
                "The agent acts as a digital marketing strategist, selecting which brand to post for, "
                "what content type to publish, when to schedule it, and (in Task 3) how to allocate "
                "paid ad budgets — optimizing engagement, conversions, and portfolio coverage over "
                "a multi-step episode."
            ),
            version="1.0.0",
        )

    def __init__(self, task_id: int = 1, seed: Optional[int] = None):
        super().__init__()
        self._task_id = task_id
        self._seed = seed
        self.reset(task_id=task_id, seed=seed)

    def reset(
        self,
        task_id: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> SocialObservation:
        """Reset the environment for a new episode."""
        if task_id is not None:
            self._task_id = task_id
        if self._task_id not in TASK_CONFIG:
            raise ValueError(f"Unsupported task_id {self._task_id}. Expected one of {sorted(TASK_CONFIG)}.")

        self._task = TASK_CONFIG[self._task_id]
        self._seed = seed if seed is not None else random.randint(0, 2**31)
        self._rng = random.Random(self._seed)

        self._data_root = None
        requested_sqlite_path = kwargs.get("sqlite_path") or kwargs.get("db_path")
        self._sqlite_path = (
            Path(requested_sqlite_path)
            if requested_sqlite_path
            else Path(os.environ.get("SOCIAL_SQLITE_PATH", DEFAULT_SQLITE_PATH))
        )
        self._brands = self._load_brands(self._task["n_brands"])
        self._max_steps = self._task["max_steps"]
        self._current_step = 0
        self._total_reward = 0.0
        self._daily_budget = self._task["daily_budget"]
        self._episode_budget_total = self._daily_budget * self._max_steps
        self._engagement_log: List[Dict[str, Any]] = []
        self._portfolio_paid_lift_total = 0.0
        self._total_policy_violations = 0
        self._total_conversions = 0.0
        self._shock_platform = self._rng.choice(["instagram", "facebook", "linkedin", "youtube", "x"])
        start_min = max(2, self._max_steps // 4)
        start_max = max(start_min + 1, self._max_steps - 3)
        self._shock_start = self._rng.randint(start_min, start_max)
        self._shock_duration = self._rng.randint(2, 4)

        for brand in self._brands:
            brand["engagement_history"] = []
            brand["recent_content_types"] = []
            brand["recent_time_slots"] = []
            brand["content_mix"] = {"reel": 0, "carousel": 0, "static": 0}
            brand["total_posts"] = 0
            brand["last_post_step"] = None
            brand["last_budget_fraction"] = 0.0
            brand["budget_spent"] = 0.0
            brand["cumulative_budget_spent"] = 0.0
            brand["budget_remaining"] = 0.0
            brand["fatigue_score"] = 0.0
            brand["trend_momentum"] = 0.0
            brand["risk_level"] = 0.0
            brand["policy_violations"] = 0
            brand["cumulative_conversions"] = 0.0
            brand["pending_conversion_queue"] = []

        self._state = SocialState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            task_id=self._task_id,
            total_reward=0.0,
            brand_configs=[{k: v for k, v in brand.items()} for brand in self._brands],
            engagement_log=[],
            total_policy_violations=0,
            total_conversions=0.0,
        )

        return self._make_observation()

    def step(self, action: SocialAction) -> SocialObservation:
        """Execute one planning step."""
        error = self._validate_action(action)
        if error:
            return self._handle_invalid_action(action, error)

        brand = self._brands[action.brand_id]
        day_of_week = self._current_step % 7
        budget_fractions = self._normalized_budget_fractions(action)
        market_trend = self._market_trend_for_step(self._current_step)
        shock_platform = self._active_shock_platform(self._current_step)

        realized_conversions = self._apply_conversion_pipeline()
        self._total_conversions += realized_conversions

        if self._task_id == 3:
            self._apply_budget_allocation(budget_fractions)
            portfolio_paid_lift = self._compute_portfolio_paid_lift(budget_fractions)
            self._portfolio_paid_lift_total += portfolio_paid_lift
        else:
            portfolio_paid_lift = 0.0
            self._clear_budget_allocation()

        days_since_last_post = self._days_since_last_post(brand)
        repeated_content_streak = self._recent_streak(
            brand["recent_content_types"], action.content_type
        )
        repeated_time_streak = self._recent_streak(
            brand["recent_time_slots"], action.post_time_slot
        )
        prior_average_engagement = self._recent_average_engagement(brand)

        engagement = compute_engagement(
            audience_type=brand["audience_type"],
            content_type=action.content_type,
            time_slot=action.post_time_slot,
            day_of_week=day_of_week,
            budget_fraction=budget_fractions[action.brand_id],
            brand_quality=brand["brand_quality"],
            days_since_last_post=days_since_last_post,
            repeated_content_streak=repeated_content_streak,
            repeated_time_streak=repeated_time_streak,
            prior_average_engagement=prior_average_engagement,
            rng=self._rng,
        )
        shock_mult = self._shock_penalty_multiplier(brand.get("platform", ""), shock_platform)
        trend_mult = 1.0 + 0.20 * market_trend
        engagement = max(0.0, min(1.0, engagement * shock_mult * trend_mult))

        policy_risk, policy_violation = self._policy_risk(brand, action)
        if policy_violation:
            brand["policy_violations"] += 1
            self._total_policy_violations += 1

        brand["engagement_history"].append(round(engagement, 4))
        brand["recent_content_types"].append(action.content_type)
        brand["recent_time_slots"].append(action.post_time_slot)
        brand["recent_content_types"] = brand["recent_content_types"][-5:]
        brand["recent_time_slots"] = brand["recent_time_slots"][-5:]
        brand["content_mix"][action.content_type] += 1
        brand["total_posts"] += 1
        brand["last_post_step"] = self._current_step
        brand["fatigue_score"] = self._fatigue_score(brand)
        brand["risk_level"] = policy_risk

        expected_conversion = self._queue_future_conversions(
            brand=brand,
            engagement=engagement,
            budget_fraction=budget_fractions[action.brand_id],
            market_trend=market_trend,
        )
        brand["trend_momentum"] = round(market_trend, 4)

        reward = self._compute_reward(
            engagement=engagement,
            action=action,
            portfolio_paid_lift=portfolio_paid_lift,
            budget_fractions=budget_fractions,
            realized_conversions=realized_conversions,
            expected_conversion=expected_conversion,
            policy_violation=policy_violation,
            market_trend=market_trend,
        )
        self._total_reward += reward

        log_entry = {
            "step": self._current_step,
            "day_of_week": day_of_week,
            "brand_id": action.brand_id,
            "content_type": action.content_type,
            "time_slot": action.post_time_slot,
            "engagement": round(engagement, 4),
            "reward": round(reward, 4),
            "portfolio_paid_lift": round(portfolio_paid_lift, 4),
            "realized_conversions": round(realized_conversions, 4),
            "expected_conversion": round(expected_conversion, 4),
            "policy_violation": policy_violation,
            "policy_risk": round(policy_risk, 4),
            "market_trend": round(market_trend, 4),
            "shock_platform": shock_platform,
            "budget_fractions": [round(fraction, 4) for fraction in budget_fractions],
            "invalid_action": False,
        }
        self._engagement_log.append(log_entry)

        self._current_step += 1
        done = self._current_step >= self._max_steps
        self._refresh_state()

        obs = self._make_observation(reward=reward, done=done)
        if done:
            self._attach_episode_metadata(obs)
        return obs

    @property
    def state(self) -> SocialState:
        return self._state

    def _validate_action(self, action: SocialAction) -> Optional[str]:
        """Validate an action. Returns an error string or `None`."""
        n_brands = self._task["n_brands"]

        if action.brand_id < 0 or action.brand_id >= n_brands:
            return f"Invalid brand_id {action.brand_id}. Must be 0-{n_brands - 1}."

        if action.content_type not in VALID_CONTENT_TYPES:
            return (
                f"Invalid content_type '{action.content_type}'. "
                f"Must be one of {sorted(VALID_CONTENT_TYPES)}."
            )

        if action.post_time_slot < 0 or action.post_time_slot > 5:
            return f"Invalid post_time_slot {action.post_time_slot}. Must be 0-5."

        if self._task_id == 3:
            if not action.budget_fractions:
                return "Task 3 requires budget_fractions (list of floats summing to 1.0)."
            if len(action.budget_fractions) != n_brands:
                return (
                    f"budget_fractions must have {n_brands} entries, "
                    f"got {len(action.budget_fractions)}."
                )
            if any(fraction < 0 for fraction in action.budget_fractions):
                return "budget_fractions cannot contain negative values."
            total = sum(action.budget_fractions)
            if abs(total - 1.0) > 0.05:
                return f"budget_fractions must sum to 1.0, got {total:.3f}."

        return None

    def _handle_invalid_action(
        self, action: SocialAction, error: str
    ) -> SocialObservation:
        """Count invalid actions as wasted days so episodes cannot stall forever."""
        penalty = -0.2 if self._task_id == 3 else -0.15
        self._total_reward += penalty

        self._engagement_log.append(
            {
                "step": self._current_step,
                "day_of_week": self._current_step % 7,
                "brand_id": action.brand_id,
                "content_type": action.content_type,
                "time_slot": action.post_time_slot,
                "engagement": 0.0,
                "reward": round(penalty, 4),
                "portfolio_paid_lift": 0.0,
                "budget_fractions": [
                    round(fraction, 4) for fraction in self._normalized_budget_fractions(action)
                ],
                "invalid_action": True,
                "error": error,
            }
        )

        self._current_step += 1
        done = self._current_step >= self._max_steps
        self._refresh_state()

        obs = self._make_observation(reward=penalty, done=done, error=error)
        if done:
            self._attach_episode_metadata(obs)
        return obs

    def _normalized_budget_fractions(self, action: SocialAction) -> List[float]:
        """Return a safe, normalized budget vector for calculations."""
        if self._task_id != 3:
            return [0.0] * len(self._brands)

        if len(action.budget_fractions) != len(self._brands):
            equal_share = 1.0 / len(self._brands)
            return [equal_share] * len(self._brands)

        raw = [max(0.0, fraction) for fraction in action.budget_fractions]
        total = sum(raw)
        if total <= 0:
            equal_share = 1.0 / len(self._brands)
            return [equal_share] * len(self._brands)
        return [fraction / total for fraction in raw]

    def _apply_budget_allocation(self, budget_fractions: List[float]) -> None:
        """Apply one day's budget allocation across the portfolio."""
        for brand, fraction in zip(self._brands, budget_fractions):
            spend = fraction * self._daily_budget
            brand["last_budget_fraction"] = round(fraction, 4)
            brand["budget_spent"] = round(brand.get("budget_spent", 0.0) + spend, 2)
            brand["cumulative_budget_spent"] = brand["budget_spent"]
            brand["budget_remaining"] = 0.0

    def _clear_budget_allocation(self) -> None:
        """Reset budget-specific observation fields for non-budget tasks."""
        for brand in self._brands:
            brand["last_budget_fraction"] = 0.0
            brand["budget_remaining"] = 0.0

    def _compute_portfolio_paid_lift(self, budget_fractions: List[float]) -> float:
        """Estimate paid-media lift for the whole portfolio for this day."""
        total_lift = 0.0
        for brand, fraction in zip(self._brands, budget_fractions):
            total_lift += compute_paid_engagement_lift(
                audience_type=brand["audience_type"],
                budget_fraction=fraction,
                brand_quality=brand["brand_quality"],
                recent_average_engagement=self._recent_average_engagement(brand),
            )
        return total_lift

    def _compute_reward(
        self,
        engagement: float,
        action: SocialAction,
        portfolio_paid_lift: float,
        budget_fractions: List[float],
        realized_conversions: float,
        expected_conversion: float,
        policy_violation: bool,
        market_trend: float,
    ) -> float:
        """Compute a dense per-step reward that matches the task objective."""
        conversion_component = 0.55 * realized_conversions + 0.10 * expected_conversion
        compliance_penalty = 0.25 if policy_violation else 0.0
        trend_alignment_bonus = 0.03 if market_trend > 0 else 0.0

        if self._task_id == 1:
            return max(-1.0, engagement + conversion_component - compliance_penalty + trend_alignment_bonus)

        if self._task_id == 2:
            brand = self._brands[action.brand_id]
            total_followers = sum(item["follower_count"] for item in self._brands)
            follower_weight = brand["follower_count"] / max(total_followers, 1)
            post_counts = self._post_counts()
            pre_action_counts = post_counts.copy()
            pre_action_counts[action.brand_id] = max(0, pre_action_counts[action.brand_id] - 1)
            min_posts = min(pre_action_counts)
            focus_penalty = 0.04 * max(0, pre_action_counts[action.brand_id] - min_posts - 1)
            neglect_penalty = 0.02 * sum(
                1 for item in self._brands if self._days_since_last_post(item) > 4
            )
            coverage_bonus = 0.05 if pre_action_counts[action.brand_id] == min_posts else 0.0
            base = engagement * (0.75 + 0.25 * follower_weight * len(self._brands))
            return max(
                -1.0,
                base
                + conversion_component
                + coverage_bonus
                + trend_alignment_bonus
                - focus_penalty
                - neglect_penalty
                - compliance_penalty,
            )

        if self._task_id == 3:
            hhi = sum(fraction * fraction for fraction in budget_fractions)
            target_hhi = 1.0 / len(self._brands)
            diversification_penalty = max(0.0, hhi - target_hhi) * 0.45
            neglected_budget_penalty = 0.015 * sum(
                1
                for brand, fraction in zip(self._brands, budget_fractions)
                if fraction < 0.05 and self._days_since_last_post(brand) > 5
            )
            # Penalize organic neglect: brands that haven't been posted for get a penalty
            organic_neglect_penalty = 0.02 * sum(
                1
                for brand in self._brands
                if self._days_since_last_post(brand) > 3 and brand["total_posts"] == 0
            )
            organic_component = 0.65 * engagement
            portfolio_component = 0.35 * (portfolio_paid_lift / len(self._brands))
            return max(
                -1.0,
                organic_component
                + portfolio_component
                + conversion_component
                - diversification_penalty
                - neglected_budget_penalty
                - organic_neglect_penalty,
                - compliance_penalty
            )

        return max(-1.0, engagement + conversion_component - compliance_penalty)

    def _compute_grader_score(self) -> float:
        """Compute the final grader score in the `[0, 1]` range."""
        if self._task_id == 1:
            engagements = [
                log["engagement"] for log in self._engagement_log if not log.get("invalid_action")
            ]
            avg = sum(engagements) / max(len(engagements), 1)
            conversion_score = min(1.0, self._total_conversions / max(self._max_steps * 0.20, 1.0))
            safety_score = 1.0 - min(1.0, self._total_policy_violations / max(self._max_steps * 0.25, 1.0))
            return min(1.0, 0.65 * (avg / 0.85) + 0.20 * conversion_score + 0.15 * safety_score)

        if self._task_id == 2:
            per_brand_avg = []
            for brand in self._brands:
                history = brand["engagement_history"]
                per_brand_avg.append(sum(history) / len(history) if history else 0.0)

            post_counts = self._post_counts()
            target_posts = self._task["target_posts_per_brand"]
            engagement_score = min(1.0, (sum(per_brand_avg) / len(per_brand_avg)) / 0.75)
            conversion_score = min(1.0, self._total_conversions / max(self._max_steps * 0.25, 1.0))
            coverage_score = sum(
                min(count / max(target_posts, 1), 1.0) for count in post_counts
            ) / len(post_counts)
            balance_score = 1.0 - (
                (max(post_counts) - min(post_counts)) / max(max(post_counts), 1)
            )
            safety_score = 1.0 - min(1.0, self._total_policy_violations / max(self._max_steps * 0.20, 1.0))
            return min(
                1.0,
                0.40 * engagement_score
                + 0.20 * conversion_score
                + 0.20 * coverage_score
                + 0.10 * balance_score
                + 0.10 * safety_score,
            )

        if self._task_id == 3:
            organic_total = sum(
                log["engagement"] for log in self._engagement_log if not log.get("invalid_action")
            )
            organic_score = min(1.0, organic_total / max(self._max_steps * 0.85, 1.0))
            paid_score = min(
                1.0,
                self._portfolio_paid_lift_total / max(self._max_steps * 0.30, 1.0),
            )
            conversion_score = min(1.0, self._total_conversions / max(self._max_steps * 0.30, 1.0))

            budget_days = [0] * len(self._brands)
            diversification_scores = []
            for log in self._engagement_log:
                fractions = log.get("budget_fractions") or []
                if not fractions:
                    continue
                for index, fraction in enumerate(fractions):
                    if fraction >= 0.10:
                        budget_days[index] += 1
                hhi = sum(fraction * fraction for fraction in fractions)
                target_hhi = 1.0 / len(self._brands)
                diversification_scores.append(
                    1.0 - max(0.0, hhi - target_hhi) / max(1.0 - target_hhi, 1e-6)
                )

            budget_coverage_target = max(1, int(self._max_steps * 0.30))
            budget_coverage_score = sum(
                min(days / budget_coverage_target, 1.0) for days in budget_days
            ) / len(budget_days)
            diversification_score = (
                sum(diversification_scores) / len(diversification_scores)
                if diversification_scores
                else 0.0
            )

            # Organic posting coverage: how many brands got at least some posts?
            post_counts = self._post_counts()
            organic_target = max(1, int(self._max_steps * 0.15))  # ~3 posts per brand for 21 steps
            organic_coverage_score = sum(
                min(count / organic_target, 1.0) for count in post_counts
            ) / len(post_counts)

            # Organic posting balance: penalize heavy imbalance
            if max(post_counts) > 0:
                organic_balance_score = 1.0 - (
                    (max(post_counts) - min(post_counts)) / max(max(post_counts), 1)
                )
            else:
                organic_balance_score = 0.0

            return min(
                1.0,
                0.22 * organic_score
                + 0.16 * paid_score
                + 0.18 * conversion_score
                + 0.15 * budget_coverage_score
                + 0.10 * diversification_score
                + 0.15 * organic_coverage_score
                + 0.10 * organic_balance_score,
            )

        return 0.0

    def _make_observation(
        self,
        reward: float = 0.0,
        done: bool = False,
        error: Optional[str] = None,
    ) -> SocialObservation:
        """Build an observation from the current environment state."""
        brands = []
        for brand in self._brands:
            history = brand["engagement_history"]
            reference_share = (
                self._episode_budget_total / len(self._brands)
                if self._task_id == 3 and self._episode_budget_total > 0
                else 0.0
            )
            brands.append(
                BrandState(
                    brand_id=brand["brand_id"],
                    brand_name=brand["brand_name"],
                    platform=brand.get("platform", ""),
                    data_source=brand.get("data_source", "synthetic"),
                    audience_type=brand["audience_type"],
                    follower_count=brand["follower_count"],
                    historical_posts=brand.get("historical_posts", 0),
                    historical_average_engagement=brand.get(
                        "historical_average_engagement", 0.0
                    ),
                    engagement_history=history[-5:],
                    average_engagement=round(sum(history) / len(history), 4) if history else 0.0,
                    recent_content_types=list(brand["recent_content_types"]),
                    recent_time_slots=list(brand["recent_time_slots"]),
                    content_mix=dict(brand["content_mix"]),
                    total_posts=brand["total_posts"],
                    days_since_last_post=self._days_since_last_post(brand),
                    budget_spent=round(brand.get("budget_spent", 0.0), 2),
                    budget_remaining=round(
                        max(0.0, reference_share - brand.get("budget_spent", 0.0)), 2
                    ),
                    last_budget_fraction=round(brand.get("last_budget_fraction", 0.0), 4),
                    fatigue_score=round(brand.get("fatigue_score", 0.0), 4),
                    trend_momentum=round(brand.get("trend_momentum", 0.0), 4),
                    risk_level=round(brand.get("risk_level", 0.0), 4),
                    policy_violations=int(brand.get("policy_violations", 0)),
                    cumulative_conversions=round(brand.get("cumulative_conversions", 0.0), 4),
                    pending_conversions=round(
                        sum(item["amount"] for item in brand.get("pending_conversion_queue", [])),
                        4,
                    ),
                )
            )

        return SocialObservation(
            brands=brands,
            current_day=self._current_step % 7,
            current_step=self._current_step,
            task_id=self._task_id,
            task_name=self._task["name"],
            max_steps=self._max_steps,
            daily_budget=self._daily_budget,
            last_action_error=error,
            market_trend=round(self._market_trend_for_step(self._current_step), 4),
            shock_event_active=self._active_shock_platform(self._current_step) is not None,
            shock_platform=self._active_shock_platform(self._current_step),
            total_policy_violations=self._total_policy_violations,
            total_conversions=round(self._total_conversions, 4),
            reward=reward,
            done=done,
            metadata={
                "seed": self._seed,
                "task_name": self._task["name"],
                "steps_remaining": max(0, self._max_steps - self._current_step),
                "data_root": str(self._data_root) if self._data_root else "",
                "sqlite_path": str(self._sqlite_path) if self._sqlite_path else "",
                "data_mode": self._brands[0].get("data_source", "synthetic") if self._brands else "synthetic",
                "shock_platform": self._shock_platform,
                "shock_window": [self._shock_start, self._shock_start + self._shock_duration - 1],
            },
        )

    def _attach_episode_metadata(self, obs: SocialObservation) -> None:
        """Attach final scoring metadata when an episode completes."""
        grader_score = self._compute_grader_score()
        obs.metadata["grader_score"] = round(grader_score, 4)
        obs.metadata["episode_summary"] = {
            "total_reward": round(self._total_reward, 4),
            "steps": self._current_step,
            "task_id": self._task_id,
            "task_name": self._task["name"],
            "portfolio_paid_lift_total": round(self._portfolio_paid_lift_total, 4),
            "total_conversions": round(self._total_conversions, 4),
            "total_policy_violations": self._total_policy_violations,
        }

    def _refresh_state(self) -> None:
        """Synchronize the exported state object with in-memory data."""
        self._state = SocialState(
            episode_id=self._state.episode_id,
            step_count=self._current_step,
            task_id=self._task_id,
            total_reward=round(self._total_reward, 4),
            brand_configs=[{k: v for k, v in brand.items()} for brand in self._brands],
            engagement_log=self._engagement_log.copy(),
            total_policy_violations=self._total_policy_violations,
            total_conversions=round(self._total_conversions, 4),
            grader_score=round(self._compute_grader_score(), 4),
        )

    def _market_trend_for_step(self, step: int) -> float:
        """Compute a smooth, deterministic market trend in [-1, 1]."""
        phase = (2 * math.pi * step) / max(6, self._max_steps)
        return max(-1.0, min(1.0, math.sin(phase + (self._seed % 17) * 0.11)))

    def _active_shock_platform(self, step: int) -> Optional[str]:
        """Return the impacted platform during the temporary shock window."""
        if self._shock_start <= step < (self._shock_start + self._shock_duration):
            return self._shock_platform
        return None

    def _shock_penalty_multiplier(self, platform: str, shock_platform: Optional[str]) -> float:
        """Apply temporary reach penalty for a shocked platform."""
        if not shock_platform:
            return 1.0
        return 0.78 if platform == shock_platform else 1.0

    def _policy_risk(self, brand: Dict[str, Any], action: SocialAction) -> tuple[float, bool]:
        """Estimate compliance risk and sampled violation event."""
        base_by_content = {
            "reel": 0.07,
            "carousel": 0.05,
            "static": 0.03,
        }
        late_slot_risk = 0.02 if action.post_time_slot in (0, 5) else 0.0
        youth_risk = 0.02 if brand.get("audience_type") == "youth" and action.content_type == "reel" else 0.0
        fatigue_risk = min(0.05, 0.01 * self._recent_streak(brand.get("recent_content_types", []), action.content_type))
        risk = min(0.35, base_by_content.get(action.content_type, 0.04) + late_slot_risk + youth_risk + fatigue_risk)
        violation = self._rng.random() < risk
        return risk, violation

    def _queue_future_conversions(
        self,
        brand: Dict[str, Any],
        engagement: float,
        budget_fraction: float,
        market_trend: float,
    ) -> float:
        """Queue delayed conversions and return expected conversion signal."""
        expected = max(
            0.0,
            (0.28 * engagement + 0.20 * math.sqrt(max(0.0, budget_fraction)))
            * (0.9 + 0.1 * brand.get("brand_quality", 1.0))
            * (1.0 + 0.12 * market_trend),
        )
        chunks = [0.45 * expected, 0.35 * expected, 0.20 * expected]
        for delay, amount in enumerate(chunks, start=1):
            brand["pending_conversion_queue"].append(
                {"eta_step": self._current_step + delay, "amount": amount}
            )
        return expected

    def _apply_conversion_pipeline(self) -> float:
        """Realize conversions whose delay has elapsed across all brands."""
        realized = 0.0
        for brand in self._brands:
            queue = brand.get("pending_conversion_queue", [])
            ready = [item for item in queue if item["eta_step"] <= self._current_step]
            future = [item for item in queue if item["eta_step"] > self._current_step]
            realized_here = sum(item["amount"] for item in ready)
            brand["cumulative_conversions"] = brand.get("cumulative_conversions", 0.0) + realized_here
            brand["pending_conversion_queue"] = future
            realized += realized_here
        return realized

    def _fatigue_score(self, brand: Dict[str, Any]) -> float:
        """Simple fatigue metric combining repetition and high frequency posting."""
        content = brand.get("recent_content_types", [])
        slots = brand.get("recent_time_slots", [])
        if not content:
            return 0.0
        repeated_content = self._recent_streak(content[:-1], content[-1]) if len(content) > 1 else 0
        repeated_slot = self._recent_streak(slots[:-1], slots[-1]) if len(slots) > 1 else 0
        cadence_pressure = 1.0 if self._days_since_last_post(brand) <= 1 else 0.0
        score = 0.25 * repeated_content + 0.20 * repeated_slot + 0.25 * cadence_pressure
        return max(0.0, min(1.0, score))

    def _days_since_last_post(self, brand: Dict[str, Any]) -> int:
        """Compute how many planning steps have passed since this brand was posted."""
        last_post_step = brand.get("last_post_step")
        if last_post_step is None:
            return max(1, self._current_step + 1)
        return max(0, self._current_step - last_post_step)

    def _recent_average_engagement(self, brand: Dict[str, Any]) -> float:
        """Average the most recent engagement values for local momentum estimation."""
        history = brand.get("engagement_history", [])
        if not history:
            return 0.0
        window = history[-3:]
        return sum(window) / len(window)

    def _recent_streak(self, history: List[Any], value: Any) -> int:
        """Count consecutive repeats of `value` at the tail of `history`."""
        streak = 0
        for item in reversed(history):
            if item != value:
                break
            streak += 1
        return streak

    def _post_counts(self) -> List[int]:
        """Return cumulative post counts per brand."""
        return [brand["total_posts"] for brand in self._brands]

    def _load_brands(self, n_brands: int) -> List[Dict[str, Any]]:
        """Load brands from SQLite, auto-seeding synthetic data if needed."""
        if self._sqlite_path:
            ensure_sqlite_seeded(self._sqlite_path)
            db_brands = load_brand_channels_from_sqlite(self._sqlite_path, n_brands)
            if db_brands:
                for index, brand in enumerate(db_brands):
                    brand["brand_id"] = index
                return db_brands

        brands = generate_brands(n_brands, self._seed)
        for brand in brands:
            brand["platform"] = "generic"
            brand["data_source"] = "synthetic"
            brand["historical_posts"] = 0
            brand["historical_average_engagement"] = 0.0
            brand["historical_content_mix"] = {"reel": 0, "carousel": 0, "static": 0}
        return brands
