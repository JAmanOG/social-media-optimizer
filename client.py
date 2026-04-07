"""
Social Media Optimizer Environment Client.

Provides the EnvClient subclass for connecting to the environment server.
"""

from __future__ import annotations

from typing import Any, Dict

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

try:
    from .models import BrandState, SocialAction, SocialObservation, SocialState
except ImportError:
    from models import BrandState, SocialAction, SocialObservation, SocialState


class SocialMediaOptimizerClient(
    EnvClient[SocialAction, SocialObservation, SocialState]
):
    """
    Client for the Social Media Optimizer Environment.

    Example:
        >>> with SocialMediaOptimizerClient(base_url="http://localhost:7860") as env:
        ...     result = env.reset()
        ...     print(result.observation.brands)
        ...     action = SocialAction(brand_id=0, content_type="reel", post_time_slot=5)
        ...     result = env.step(action)
        ...     print(result.reward, result.done)
    """

    def _step_payload(self, action: SocialAction) -> Dict[str, Any]:
        return action.model_dump()

    def _parse_result(
        self, payload: Dict[str, Any]
    ) -> StepResult[SocialObservation]:
        obs_data = payload.get("observation", {})
        observation = SocialObservation(
            brands=[BrandState(**brand) for brand in obs_data.get("brands", [])],
            current_day=obs_data.get("current_day", 0),
            current_step=obs_data.get("current_step", 0),
            task_id=obs_data.get("task_id", 1),
            task_name=obs_data.get("task_name", ""),
            max_steps=obs_data.get("max_steps", 7),
            daily_budget=obs_data.get("daily_budget", 0.0),
            last_action_error=obs_data.get("last_action_error"),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> SocialState:
        return SocialState(
            episode_id=payload.get("episode_id", ""),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id", 1),
            total_reward=payload.get("total_reward", 0.0),
            brand_configs=payload.get("brand_configs", []),
            engagement_log=payload.get("engagement_log", []),
        )
