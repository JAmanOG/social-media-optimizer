"""
Data models for Social Media Optimizer Environment.

Defines the Action, Observation, and State types for the multi-brand
social media agency optimization environment.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from openenv.core.env_server import Action, Observation, State
from pydantic import BaseModel, Field


class SocialAction(Action):
    """
    Action for the Social Media Optimizer environment.

    The agent selects which brand to act on, what content type to post,
    when to schedule it, and how to allocate the ad budget.
    """

    brand_id: int = Field(
        description="Index of the brand to post for (0-indexed)."
    )
    content_type: str = Field(
        description="Type of content: 'reel', 'carousel', or 'static'."
    )
    post_time_slot: int = Field(
        description="Time slot index: 0=6AM, 1=9AM, 2=12PM, 3=3PM, 4=6PM, 5=9PM."
    )
    budget_fractions: List[float] = Field(
        default_factory=list,
        description=(
            "Per-brand budget allocation fractions (must sum to 1.0). "
            "Only used in Task 3. Empty list for Tasks 1 & 2."
        ),
    )


class BrandState(BaseModel):
    """Snapshot of a single brand's current metrics (nested in observation)."""

    model_config = {"extra": "allow"}

    brand_id: int = 0
    brand_name: str = ""
    platform: str = ""
    data_source: str = "synthetic"
    audience_type: str = ""
    follower_count: int = 0
    historical_posts: int = 0
    historical_average_engagement: float = 0.0
    engagement_history: List[float] = Field(default_factory=list)
    average_engagement: float = 0.0
    recent_content_types: List[str] = Field(default_factory=list)
    recent_time_slots: List[int] = Field(default_factory=list)
    content_mix: Dict[str, int] = Field(default_factory=dict)
    total_posts: int = 0
    days_since_last_post: int = 0
    budget_spent: float = 0.0
    budget_remaining: float = 0.0
    last_budget_fraction: float = 0.0
    fatigue_score: float = 0.0
    trend_momentum: float = 0.0
    risk_level: float = 0.0
    policy_violations: int = 0
    cumulative_conversions: float = 0.0
    pending_conversions: float = 0.0


class SocialObservation(Observation):
    """
    Observation for the Social Media Optimizer environment.

    Provides the agent with brand performance data, time context,
    and task configuration.
    """

    brands: List[BrandState] = Field(
        default_factory=list,
        description="List of per-brand state snapshots.",
    )
    current_day: int = Field(
        default=0, description="Current simulation day (0-indexed)."
    )
    current_step: int = Field(
        default=0, description="Current step within the episode."
    )
    task_id: int = Field(
        default=1, description="Active task ID (1, 2, or 3)."
    )
    max_steps: int = Field(
        default=7, description="Total steps in the episode."
    )
    daily_budget: float = Field(
        default=0.0, description="Total daily ad budget (Task 3 only)."
    )
    task_name: str = Field(default="", description="Human-readable task name.")
    last_action_error: Optional[str] = Field(
        default=None, description="Error message if the last action was invalid."
    )
    market_trend: float = Field(
        default=0.0,
        description="Global market trend multiplier in [-1, 1] affecting all brands.",
    )
    shock_event_active: bool = Field(
        default=False,
        description="Whether a temporary platform shock event is active for this step.",
    )
    shock_platform: Optional[str] = Field(
        default=None,
        description="Platform impacted by the active shock event, if any.",
    )
    total_policy_violations: int = Field(
        default=0,
        description="Cumulative policy violations across all brands in the episode.",
    )
    total_conversions: float = Field(
        default=0.0,
        description="Cumulative realized conversions across all brands.",
    )


class SocialState(State):
    """
    Internal state for the Social Media Optimizer environment.
    """

    task_id: int = 1
    total_reward: float = 0.0
    brand_configs: List[Dict[str, Any]] = Field(default_factory=list)
    engagement_log: List[Dict[str, Any]] = Field(default_factory=list)
    total_policy_violations: int = 0
    total_conversions: float = 0.0
    grader_score: float = 0.0
