#!/usr/bin/env python3
"""
Baseline inference script for the Social Media Optimizer environment.

MANDATORY env vars:
    API_BASE_URL       The API endpoint for the LLM.
    MODEL_NAME         The model identifier to use for inference.
    HF_TOKEN           Your Hugging Face / API key.
    LOCAL_IMAGE_NAME   Docker image name (only for from_docker_image mode).

STDOUT FORMAT:
    [START] task=<task_name> env=social-media-optimizer model=<model_name>
    [STEP]  step=<n> action=<json> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...,rn>
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import textwrap
from pathlib import Path
from typing import List

import requests
from openai import OpenAI


# ── .env loader ──────────────────────────────────────────────────────
def _load_local_env() -> None:
    """Lightweight `.env` loader so local runs work without extra dependencies."""
    env_path = Path(__file__).with_name(".env")
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'").strip('"'))


_load_local_env()


# ── Configuration (mandatory env vars) ───────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL","https://openrouter.ai/api/v1")
MODEL_NAME = os.getenv("MODEL_NAME","openrouter/free")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "social-media-optimizer")

ENV_PORT = int(os.getenv("ENV_PORT", "7860"))
ENV_BASE_URL = os.getenv("ENV_BASE_URL", f"http://localhost:{ENV_PORT}")
ENV_NAME = "social-media-optimizer"

TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "512"))

TASK_NAMES = {
    1: "single-brand-engagement",
    2: "multi-brand-content-mix",
    3: "budget-allocation-cross-brand",
}


# ── System prompt ────────────────────────────────────────────────────
SYSTEM_PROMPT = textwrap.dedent(
    """\
    You are an AI social media strategist for a digital marketing agency.
    You manage multiple brands and must optimize posting decisions to maximize
    engagement while keeping portfolio coverage healthy.

    On each turn you receive a JSON observation with:
    - brands: list of brand states (check days_since_last_post and total_posts)
    - current_day: 0=Mon…6=Sun
    - current_step / max_steps: progress through the episode
    - task_id / task_name
    - daily_budget: total daily ad budget (task 3 only)

    Each brand includes:
    - platform, audience_type, follower_count
    - engagement_history and average_engagement
    - recent_content_types and recent_time_slots
    - content_mix and total_posts
    - days_since_last_post (CRITICAL for scoring)
    - budget_spent and last_budget_fraction
    - fatigue_score, trend_momentum, risk_level, policy_violations
    - cumulative_conversions and pending_conversions

    Global fields include:
    - market_trend: positive means aggressive growth window, negative means defensive period
    - shock_event_active / shock_platform: temporary platform-level performance suppression
    - total_policy_violations and total_conversions

    Respond with ONLY a JSON object (no markdown, no explanation):
    {
      "brand_id": <int>,
      "content_type": "<reel|carousel|static>",
      "post_time_slot": <0-5>,
      "budget_fractions": [<float>, ...]
    }

    Time slots: 0=6AM, 1=9AM, 2=12PM, 3=3PM, 4=6PM, 5=9PM.

    CRITICAL STRATEGY RULES:
    1. ROTATE BRANDS. Every brand must get posts. Choose the brand with the
       highest days_since_last_post or lowest total_posts FIRST. Neglecting
       any brand dramatically reduces your final grader score.
    2. Match content to audience: youth/consumer → reel at slots 4-5,
       b2b → carousel at slots 1-2, general → reel at slots 3-4.
    3. Avoid repeating the same content_type or time_slot for the same brand
       on consecutive turns — it causes engagement fatigue.
    4. In Task 3, spread budget_fractions across ALL brands (e.g. [0.2,0.2,0.2,0.2,0.2]).
       Never give 0.0 to any brand. The grader rewards diversification.
    5. In Tasks 2 and 3, aim for roughly equal total_posts across all brands.
     6. During shock_event_active, de-prioritize brands on the shock_platform.
     7. Keep policy risk low: avoid repeatedly selecting high risk brands with high fatigue_score.
     8. Optimize delayed outcomes: prefer brands with strong pending_conversions momentum and low violations.
    """
)


# ── Helpers ──────────────────────────────────────────────────────────
def _create_local_env():
    """Import and create the environment directly (no HTTP server)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from server.social_media_environment import SocialMediaOptimizerEnv
    from models import SocialAction

    return SocialMediaOptimizerEnv, SocialAction


def parse_action(response_text: str, task_id: int, n_brands: int) -> dict:
    """Parse the model response into a safe action dictionary."""
    text = response_text.strip()
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text)

    try:
        action = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                action = json.loads(match.group())
            except json.JSONDecodeError:
                action = {}
        else:
            action = {}

    action.setdefault("brand_id", 0)
    action.setdefault("content_type", "reel")
    action.setdefault("post_time_slot", 5)

    if task_id == 3:
        raw_budget = action.get("budget_fractions", [])
        if not isinstance(raw_budget, list) or len(raw_budget) != n_brands:
            action["budget_fractions"] = [1.0 / n_brands] * n_brands
        else:
            cleaned = [max(0.0, float(v)) for v in raw_budget]
            total = sum(cleaned)
            action["budget_fractions"] = (
                [v / total for v in cleaned]
                if total > 0
                else [1.0 / n_brands] * n_brands
            )
    else:
        action["budget_fractions"] = []

    action["brand_id"] = max(0, min(int(action.get("brand_id", 0)), n_brands - 1))
    action["post_time_slot"] = max(0, min(int(action.get("post_time_slot", 5)), 5))
    if action["content_type"] not in ("reel", "carousel", "static"):
        action["content_type"] = "reel"

    return action


def _action_str(action: dict) -> str:
    """Compact action string for [STEP] output."""
    return json.dumps(action, separators=(",", ":"))


def _completion(client: OpenAI, messages: list[dict]) -> str:
    """Generate one chat completion."""
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        stream=False,
    )
    return completion.choices[0].message.content or ""


def _check_server(env_url: str) -> bool:
    """Check if the HTTP server is running."""
    try:
        return requests.get(f"{env_url}/health", timeout=3).status_code == 200
    except Exception:
        return False


# ── Task runners ─────────────────────────────────────────────────────
def run_task_local(client: OpenAI, task_id: int) -> float:
    """Run one task in-process with mandatory [START]/[STEP]/[END] output."""
    try:
        EnvClass, ActionClass = _create_local_env()
    except Exception as exc:
        task_name = TASK_NAMES.get(task_id, f"task-{task_id}")
        print(f"[START] task={task_name} env={ENV_NAME} model={MODEL_NAME}")
        print(
            f"[STEP] step=0 action={{}} reward=0.00 done=true "
            f"error=Local env unavailable: {exc}",
            file=sys.stderr,
        )
        print(f"[END] success=false steps=0 score=0.00 rewards=")
        return 0.0

    task_name = TASK_NAMES.get(task_id, f"task-{task_id}")

    env = EnvClass(task_id=task_id, seed=42)
    obs = env.reset(task_id=task_id, seed=42)

    n_brands = len(obs.brands)
    max_steps = obs.max_steps

    # [START]
    print(f"[START] task={task_name} env={ENV_NAME} model={MODEL_NAME}")

    rewards: List[float] = []
    step_num = 0
    success = False
    score = 0.0

    try:
        for step in range(max_steps):
            step_num = step + 1
            obs_summary = json.dumps(obs.model_dump(), indent=2)[:2400]
            user_content = f"Step {step_num}/{max_steps}. Current observation:\n{obs_summary}"

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]

            response_text = _completion(client, messages)

            action_dict = parse_action(response_text, task_id, n_brands)

            obs = env.step(ActionClass(**action_dict))
            reward = obs.reward if obs.reward is not None else 0.0
            rewards.append(reward)
            done = obs.done
            error = obs.last_action_error

            # [STEP]
            error_str = error if error else "null"
            done_str = "true" if done else "false"
            print(
                f"[STEP] step={step_num} "
                f"action={_action_str(action_dict)} "
                f"reward={reward:.2f} "
                f"done={done_str} "
                f"error={error_str}"
            )

            if done:
                score = obs.metadata.get("grader_score", 0.0)
                success = True
                break

        if not success:
            score = obs.metadata.get("grader_score", 0.0)
            success = score > 0.0

    except Exception as exc:
        print(f"[STEP] step={step_num} action={{}} reward=0.00 done=true error={exc}", file=sys.stderr)
        success = False

    # [END]
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    success_str = "true" if success else "false"
    print(
        f"[END] success={success_str} "
        f"steps={step_num} "
        f"score={score:.2f} "
        f"rewards={rewards_str}"
    )

    return score


def run_task_ws(client: OpenAI, env_url: str, task_id: int) -> float:
    """Run one task via WebSocket (stateful session) with mandatory [START]/[STEP]/[END] output."""
    try:
        from client import SocialMediaOptimizerClient
        from models import SocialAction as WSSocialAction
    except ImportError as exc:
        # WebSocket client unavailable — fall back to local execution
        return run_task_local(client, task_id)

    task_name = TASK_NAMES.get(task_id, f"task-{task_id}")
    rewards: List[float] = []
    step_num = 0
    success = False
    score = 0.0

    async def _run() -> None:
        nonlocal step_num, success, score, rewards

        try:
            async with SocialMediaOptimizerClient(base_url=env_url) as env:
                try:
                    result = await env.reset(seed=42, task_id=task_id)
                except Exception as exc:
                    print(f"[START] task={task_name} env={ENV_NAME} model={MODEL_NAME}")
                    print(
                        f"[STEP] step=0 action={{}} reward=0.00 done=true "
                        f"error=Reset failed: {exc}",
                        file=sys.stderr,
                    )
                    print(f"[END] success=false steps=0 score=0.00 rewards=")
                    return

                obs = result.observation
                n_brands = len(obs.brands)
                max_steps = obs.max_steps

                print(f"[START] task={task_name} env={ENV_NAME} model={MODEL_NAME}")

                try:
                    for step in range(max_steps):
                        step_num = step + 1
                        obs_summary = json.dumps(obs.model_dump(), indent=2)[:2400]
                        user_content = (
                            f"Step {step_num}/{max_steps}. Current observation:\n{obs_summary}"
                        )

                        messages = [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_content},
                        ]

                        try:
                            response_text = _completion(client, messages)
                        except Exception as exc:
                            response_text = ""

                        action_dict = parse_action(response_text, task_id, n_brands)

                        result = await env.step(WSSocialAction(**action_dict))
                        obs = result.observation
                        reward = obs.reward if obs.reward is not None else 0.0
                        rewards.append(reward)
                        done = obs.done
                        error = obs.last_action_error

                        error_str = error if error else "null"
                        done_str = "true" if done else "false"
                        print(
                            f"[STEP] step={step_num} "
                            f"action={_action_str(action_dict)} "
                            f"reward={reward:.2f} "
                            f"done={done_str} "
                            f"error={error_str}"
                        )

                        if done:
                            # Get grader_score from server state
                            try:
                                state = await env.state()
                                score = state.grader_score
                            except Exception:
                                score = obs.metadata.get("grader_score", 0.0)
                            success = True
                            break

                    if not success:
                        try:
                            state = await env.state()
                            score = state.grader_score
                        except Exception:
                            score = obs.metadata.get("grader_score", 0.0)
                        success = score > 0.0

                except Exception as exc:
                    print(
                        f"[STEP] step={step_num} action={{}} reward=0.00 done=true error={exc}",
                        file=sys.stderr,
                    )
                    success = False

        except Exception as exc:
            print(f"[START] task={task_name} env={ENV_NAME} model={MODEL_NAME}")
            print(
                f"[STEP] step=0 action={{}} reward=0.00 done=true "
                f"error=WebSocket connection failed: {exc}",
                file=sys.stderr,
            )
            print(f"[END] success=false steps=0 score=0.00 rewards=")
            return

        rewards_str = ",".join(f"{r:.2f}" for r in rewards)
        success_str = "true" if success else "false"
        print(
            f"[END] success={success_str} "
            f"steps={step_num} "
            f"score={score:.2f} "
            f"rewards={rewards_str}"
        )

    try:
        asyncio.run(_run())
    except RuntimeError:
        # Already-running event loop (e.g. Jupyter) — use local fallback
        return run_task_local(client, task_id)

    return score


# ── Main ─────────────────────────────────────────────────────────────
def main() -> None:
    """Run inference against all three tasks."""
    missing = []
    if not API_BASE_URL:
        missing.append("API_BASE_URL")
    if not MODEL_NAME:
        missing.append("MODEL_NAME")
    if not HF_TOKEN:
        missing.append("HF_TOKEN")
    if missing:
        raise RuntimeError(f"Missing mandatory env vars: {', '.join(missing)}")

    llm_client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    use_ws = _check_server(ENV_BASE_URL)

    for task_id in (1, 2, 3):
        try:
            if use_ws:
                run_task_ws(llm_client, ENV_BASE_URL, task_id)
            else:
                run_task_local(llm_client, task_id)
        except Exception as exc:
            task_name = TASK_NAMES.get(task_id, f"task-{task_id}")
            print(f"[START] task={task_name} env={ENV_NAME} model={MODEL_NAME}")
            print(
                f"[STEP] step=0 action={{}} reward=0.00 done=true error={exc}",
                file=sys.stderr,
            )
            print(f"[END] success=false steps=0 score=0.00 rewards=")


if __name__ == "__main__":
    main()
