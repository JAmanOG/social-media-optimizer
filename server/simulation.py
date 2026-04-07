"""
Synthetic engagement simulator for the Social Media Optimizer.

Models time-of-day effects, content affinity, and budget multipliers
deterministically (seeded random) so graders are reproducible.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List


# Time slots: 6AM, 9AM, 12PM, 3PM, 6PM, 9PM
TIME_SLOT_LABELS = ["6AM", "9AM", "12PM", "3PM", "6PM", "9PM"]

# Base engagement curves per audience type (indexed by time slot 0-5)
# Higher values = more engagement at that time
AUDIENCE_CURVES = {
    "consumer": [0.10, 0.20, 0.35, 0.30, 0.70, 0.90],
    "b2b":      [0.15, 0.80, 0.60, 0.50, 0.25, 0.10],
    "youth":    [0.05, 0.15, 0.40, 0.45, 0.65, 0.95],
    "general":  [0.15, 0.30, 0.45, 0.40, 0.55, 0.70],
}

# Content affinity: how well each content type performs per audience type
CONTENT_AFFINITY = {
    "consumer": {"reel": 0.85, "carousel": 0.55, "static": 0.30},
    "b2b":      {"reel": 0.40, "carousel": 0.80, "static": 0.65},
    "youth":    {"reel": 0.95, "carousel": 0.45, "static": 0.15},
    "general":  {"reel": 0.70, "carousel": 0.60, "static": 0.45},
}


def compute_engagement(
    audience_type: str,
    content_type: str,
    time_slot: int,
    day_of_week: int,
    budget_fraction: float,
    brand_quality: float,
    days_since_last_post: int,
    repeated_content_streak: int,
    repeated_time_streak: int,
    prior_average_engagement: float,
    rng: random.Random,
) -> float:
    """
    Compute engagement rate (0.0–1.0) for a single post.

    Args:
        audience_type: One of 'consumer', 'b2b', 'youth', 'general'.
        content_type: One of 'reel', 'carousel', 'static'.
        time_slot: Index 0-5 (6AM to 9PM).
        day_of_week: 0=Mon ... 6=Sun.
        budget_fraction: Fraction of daily budget spent on this brand (0.0–1.0).
        brand_quality: Baseline brand quality factor (0.5–1.5).
        days_since_last_post: Days since the brand last posted.
        repeated_content_streak: Number of recent consecutive posts using the same content type.
        repeated_time_streak: Number of recent consecutive posts using the same slot.
        prior_average_engagement: Mean of the recent engagement history.
        rng: Seeded random.Random instance for reproducibility.

    Returns:
        Engagement rate in [0.0, 1.0].
    """
    # Base time-of-day engagement
    curve = AUDIENCE_CURVES.get(audience_type, AUDIENCE_CURVES["general"])
    time_score = curve[min(time_slot, 5)]

    # Content affinity
    affinity = CONTENT_AFFINITY.get(audience_type, CONTENT_AFFINITY["general"])
    content_score = affinity.get(content_type, 0.3)

    # Weekend boost for consumer/youth, weekday boost for b2b
    weekend = day_of_week >= 5
    if weekend and audience_type in ("consumer", "youth"):
        day_modifier = 1.15
    elif not weekend and audience_type == "b2b":
        day_modifier = 1.10
    else:
        day_modifier = 1.0

    # Budget multiplier (diminishing returns)
    budget_mult = 1.0 + 0.5 * math.sqrt(max(budget_fraction, 0.0))

    # Posting cadence: daily repetition hurts, short healthy gaps help, long silence hurts.
    if days_since_last_post <= 0:
        cadence_mult = 0.88
    elif days_since_last_post == 1:
        cadence_mult = 1.0
    elif days_since_last_post <= 3:
        cadence_mult = 1.08
    elif days_since_last_post <= 5:
        cadence_mult = 1.02
    else:
        cadence_mult = 0.92

    # Repeating the same creative/time pattern creates fatigue.
    content_fatigue_mult = max(0.78, 1.0 - 0.08 * repeated_content_streak)
    time_fatigue_mult = max(0.82, 1.0 - 0.05 * repeated_time_streak)

    # Small momentum term so good recent decisions create some persistence without dominating.
    momentum_mult = 0.95 + min(0.10, max(prior_average_engagement, 0.0) * 0.18)

    # Combine factors
    raw = (
        time_score
        * content_score
        * day_modifier
        * budget_mult
        * cadence_mult
        * content_fatigue_mult
        * time_fatigue_mult
        * momentum_mult
        * brand_quality
    )

    # Add small noise (seeded)
    noise = rng.gauss(0, 0.03)
    raw += noise

    # Clamp to [0, 1]
    return max(0.0, min(1.0, raw))


def compute_paid_engagement_lift(
    audience_type: str,
    budget_fraction: float,
    brand_quality: float,
    recent_average_engagement: float,
) -> float:
    """
    Estimate paid-media lift from one day's budget allocation.

    The output is intentionally smaller than organic engagement and uses
    diminishing returns so that concentrating the full budget on one brand
    is usually suboptimal.
    """
    base_response = {
        "consumer": 0.34,
        "b2b": 0.28,
        "youth": 0.36,
        "general": 0.30,
    }.get(audience_type, 0.30)

    frac = max(0.0, min(1.0, budget_fraction))
    diminishing_return = math.sqrt(frac)
    saturation_penalty = max(0.0, frac - 0.45) * 0.35
    recent_signal = 0.92 + min(0.12, max(recent_average_engagement, 0.0) * 0.16)

    raw = base_response * diminishing_return * recent_signal * brand_quality
    raw *= 1.0 - saturation_penalty
    return max(0.0, min(0.45, raw))


def generate_brand(brand_id: int, audience_type: str, rng: random.Random) -> Dict:
    """
    Generate a brand configuration dict.

    Args:
        brand_id: Numeric ID.
        audience_type: Type of audience.
        rng: Seeded RNG.

    Returns:
        Brand configuration dictionary.
    """
    names = [
        "StyleHaus", "TechVista", "FreshBites", "UrbanEdge", "GlowUp",
        "DataPulse", "WildRoam", "NeonDrift", "CoreFit", "BrightPath",
        "PixelForge", "VelvetLux", "AquaPure", "BloomCo", "NovaTrend",
    ]
    name = names[brand_id % len(names)]
    followers = rng.randint(5_000, 500_000)
    quality = round(rng.uniform(0.6, 1.3), 2)

    return {
        "brand_id": brand_id,
        "brand_name": f"{name}_{brand_id}",
        "audience_type": audience_type,
        "follower_count": followers,
        "brand_quality": quality,
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


def generate_brands(
    n_brands: int, seed: int
) -> List[Dict]:
    """
    Generate a list of brand configs for an episode.

    Args:
        n_brands: Number of brands to generate.
        seed: Random seed for reproducibility.

    Returns:
        List of brand config dicts.
    """
    rng = random.Random(seed)
    audience_types = list(AUDIENCE_CURVES.keys())
    brands = []
    for i in range(n_brands):
        atype = audience_types[i % len(audience_types)]
        brands.append(generate_brand(i, atype, rng))
    return brands
