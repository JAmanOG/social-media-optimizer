---
title: Social Media Optimizer
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - social-media
license: apache-2.0
---

# Social Media Optimizer

`Social Media Optimizer` is an OpenEnv reinforcement learning environment where an
agent acts like a strategist inside a multi-brand social media agency.

Instead of optimizing a single metric in one step, the agent has to make
repeated planning decisions across an episode:

- which brand to post for
- what content type to publish
- when to publish it
- how to allocate paid budget in Task 3

The environment is designed to reward long-horizon decisions, not just short-term
engagement spikes.

## What This Environment Simulates

Each step represents one planning day. The environment tracks:

- recent engagement and average engagement
- content fatigue from repetitive choices
- posting neglect across brands
- delayed conversions
- policy and compliance risk
- market trend shifts
- temporary platform shock events
- paid-media lift in the budget allocation task

All runtime data is synthetic, so the project is safe to run and share.

## At A Glance

| Item | Details |
| --- | --- |
| Environment name | `social_media_optimizer` |
| Runtime | FastAPI + OpenEnv |
| Default port | `7860` |
| Action fields | `brand_id`, `content_type`, `post_time_slot`, `budget_fractions` |
| Content types | `reel`, `carousel`, `static` |
| Time slots | `0=6AM`, `1=9AM`, `2=12PM`, `3=3PM`, `4=6PM`, `5=9PM` |
| Tasks | 3 |
| Data mode | synthetic-only |

## Task Guide

### Task 1: Single Brand Engagement

Use one brand over 7 steps.

Primary objective:
- maximize engagement quality
- avoid repetitive fatigue
- keep policy violations low

Task configuration:
- brands: 1
- episode length: 7
- daily budget: `0.0`
- `budget_fractions`: not used

### Task 2: Multi-Brand Content Mix

Use three brands over 14 steps.

Primary objective:
- avoid neglecting any brand
- balance posting coverage
- keep engagement solid across the portfolio

Task configuration:
- brands: 3
- episode length: 14
- daily budget: `0.0`
- `budget_fractions`: not used

### Task 3: Budget Allocation + Cross-Brand

Use five brands over 21 steps while making both organic and paid decisions.

Primary objective:
- keep organic performance strong
- distribute paid budget intelligently
- diversify budget instead of over-concentrating it
- avoid leaving brands neglected for too long

Task configuration:
- brands: 5
- episode length: 21
- daily budget: `1000.0`
- `budget_fractions`: required

## Action Format

Every action follows this schema:

```json
{
  "brand_id": 0,
  "content_type": "reel",
  "post_time_slot": 5,
  "budget_fractions": []
}
```

Field meanings:

- `brand_id`: zero-based index of the brand you are posting for
- `content_type`: one of `reel`, `carousel`, `static`
- `post_time_slot`: integer from `0` to `5`
- `budget_fractions`: list of floats used only in Task 3

### Task-Specific Action Rules

- Task 1: `brand_id` must be `0`
- Task 2: `brand_id` must be `0`, `1`, or `2`
- Task 3: `brand_id` must be `0` through `4`
- Task 1 and Task 2: leave `budget_fractions` empty or set it to `[]`
- Task 3: `budget_fractions` must contain exactly 5 floats
- Task 3: `budget_fractions` must be non-negative
- Task 3: `budget_fractions` must sum to about `1.0`

### Valid Examples

Task 1 example:

```json
{
  "brand_id": 0,
  "content_type": "reel",
  "post_time_slot": 5,
  "budget_fractions": []
}
```

Task 2 example:

```json
{
  "brand_id": 2,
  "content_type": "carousel",
  "post_time_slot": 2,
  "budget_fractions": []
}
```

Task 3 example:

```json
{
  "brand_id": 1,
  "content_type": "reel",
  "post_time_slot": 4,
  "budget_fractions": [0.2, 0.2, 0.2, 0.2, 0.2]
}
```

## Observation Guide

Each observation contains both portfolio-level state and per-brand state.

Top-level fields:

- `brands`: list of current brand snapshots
- `current_day`: day-of-week index
- `current_step`: current step in the episode
- `task_id`: current task number
- `task_name`: human-readable task name
- `max_steps`: episode length
- `daily_budget`: daily paid budget for Task 3
- `last_action_error`: validation error for the previous action, if any
- `market_trend`: current trend signal in `[-1, 1]`
- `shock_event_active`: whether a platform shock is active now
- `shock_platform`: which platform is affected during the shock window
- `total_policy_violations`: cumulative violations so far
- `total_conversions`: cumulative realized conversions so far
- `metadata`: extra episode metadata such as seed, data mode, and shock window

Per-brand fields include:

- `brand_name`
- `platform`
- `audience_type`
- `follower_count`
- `engagement_history`
- `average_engagement`
- `recent_content_types`
- `recent_time_slots`
- `content_mix`
- `total_posts`
- `days_since_last_post`
- `budget_spent`
- `budget_remaining`
- `last_budget_fraction`
- `fatigue_score`
- `trend_momentum`
- `risk_level`
- `policy_violations`
- `cumulative_conversions`
- `pending_conversions`

## What The Environment Rewards

The reward is dense, but the final `grader_score` is the most useful summary of
episode quality.

High-level scoring behavior:

- Task 1 rewards strong engagement, conversions, and safety
- Task 2 rewards engagement plus brand coverage and balance
- Task 3 rewards organic engagement, paid lift, diversification, conversion
  quality, and balanced brand support

The environment penalizes:

- invalid actions
- repeated content or slot choices that create fatigue
- policy violations
- over-focusing on one brand while others are neglected
- over-concentrating budget in Task 3

Important practical point:

- Invalid actions still consume a step, so they are not free retries.

## Quick Start

### Prerequisites

- Python `3.10+`
- `uv`

Install `uv` if needed:

```bash
pip install uv
```

### 1. Clone And Install

```bash
git clone <your-repo-url>
cd social-media-optimizer
uv sync
```

### 2. Initialize The Synthetic SQLite Database

```bash
uv run python init_sqlite_db.py
```

This creates or refreshes `data/social_media.db`.

### 3. Run The Smoke Test

```bash
$env:SOCIAL_SQLITE_PATH="data/social_media.db"; uv run python test_smoke.py
```

Bash version:

```bash
SOCIAL_SQLITE_PATH=./data/social_media.db uv run python test_smoke.py
```

### 4. Start The Environment Server

```bash
uv run uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### 5. Open The Web Interface

Once the server is running locally, open:

```text
http://localhost:7860/web
```

Health endpoint:

```text
http://localhost:7860/health
```

## Using The OpenEnv Web Interface

The `/web` interface is the easiest way to test actions manually.

Typical flow:

- reset the environment with the `task_id` you want to test
- inspect the returned observation
- submit a step action
- repeat until `done=true`

### Web UI Input Tips

- `brand_id` is zero-based
- `content_type` must be exactly `reel`, `carousel`, or `static`
- `post_time_slot` must be an integer from `0` to `5`
- for Task 1 and Task 2, leave `budget_fractions` empty or use `[]`
- for Task 3, enter a JSON-style list string such as `[0.2, 0.2, 0.2, 0.2, 0.2]`

### Common Web UI Mistakes

- Using `brand_id=1` in Task 1. There is only one brand there, so the valid id is `0`.
- Entering only 3 budget fractions in Task 3. Task 3 always needs 5 values.
- Entering fractions that do not sum to about `1.0`.
- Using an unsupported content type like `video` instead of `reel`.

### Example Web UI Inputs

Task 1:

- Brand Id: `0`
- Content Type: `reel`
- Post Time Slot: `5`
- Budget Fractions: `[]`

Task 2:

- Brand Id: `1`
- Content Type: `carousel`
- Post Time Slot: `2`
- Budget Fractions: `[]`

Task 3:

- Brand Id: `1`
- Content Type: `reel`
- Post Time Slot: `5`
- Budget Fractions: `[0.2, 0.2, 0.2, 0.2, 0.2]`

## Using The Python Client

You can drive the environment programmatically with the included client.

```python
from client import SocialMediaOptimizerClient
from models import SocialAction

with SocialMediaOptimizerClient(base_url="http://localhost:7860") as env:
    result = env.reset(task_id=2, seed=42)
    print(result.observation.task_name)

    step_result = env.step(
        SocialAction(
            brand_id=1,
            content_type="carousel",
            post_time_slot=2,
            budget_fractions=[],
        )
    )

    print(step_result.reward, step_result.done)
```

Task 3 client example:

```python
from client import SocialMediaOptimizerClient
from models import SocialAction

with SocialMediaOptimizerClient(base_url="http://localhost:7860") as env:
    env.reset(task_id=3, seed=42)
    result = env.step(
        SocialAction(
            brand_id=0,
            content_type="reel",
            post_time_slot=4,
            budget_fractions=[0.2, 0.2, 0.2, 0.2, 0.2],
        )
    )
    print(result.observation.total_conversions)
```

## Running The Baseline Inference Script

`inference.py` runs a simple LLM baseline across all three tasks.

The script uses the OpenAI-compatible client interface, so you configure it with:

- `API_BASE_URL`: API base URL for your LLM provider
- `MODEL_NAME`: model identifier
- `HF_TOKEN`: API key used by the OpenAI-compatible client
- `LLM_TEMPERATURE`: optional, default `0.3`
- `LLM_MAX_TOKENS`: optional, default `512`

Notes:

- `HF_TOKEN` is currently the env var name used by the script for the API key
  even if your provider is not Hugging Face.
- `inference.py` automatically loads a local `.env` file if one exists.
- If the environment server is already running, the script will use it.
- If the server is not reachable, the script falls back to a local in-process run.

### Example: OpenRouter

PowerShell:

```powershell
$env:API_BASE_URL="https://openrouter.ai/api/v1"
$env:MODEL_NAME="openai/gpt-4.1-mini"
$env:HF_TOKEN="<your_api_key>"
$env:SOCIAL_SQLITE_PATH="data/social_media.db"
uv run python inference.py
```

Bash:

```bash
API_BASE_URL=https://openrouter.ai/api/v1 \
MODEL_NAME=openai/gpt-4.1-mini \
HF_TOKEN=<your_api_key> \
SOCIAL_SQLITE_PATH=./data/social_media.db \
uv run python inference.py
```

## Docker

Build the image:

```bash
docker build -t social-media-optimizer .
```

Run the container:

```bash
docker run -p 7860:7860 social-media-optimizer
```

Then open:

```text
http://localhost:7860/web
```

## Deploy To Hugging Face Spaces

### Option 1: OpenEnv CLI

```bash
uv run openenv push . --repo-id <username>/social-media-optimizer --exclude .openenvignore
```

### Option 2: Helper Script

```bash
bash push_space.sh <username>/social-media-optimizer
```

What gets deployed:

- FastAPI app from `server.app:app`
- OpenEnv metadata from `openenv.yaml`
- project README as the environment docs

## Project Layout

```text
social-media-optimizer/
|-- server/
|   |-- app.py
|   |-- data_source.py
|   |-- simulation.py
|   `-- social_media_environment.py
|-- data/
|   `-- social_media.db
|-- client.py
|-- inference.py
|-- init_sqlite_db.py
|-- models.py
|-- openenv.yaml
|-- push_space.sh
|-- pyproject.toml
|-- test_smoke.py
`-- README.md
```

File roles:

- `server/social_media_environment.py`: reward logic, validation, observations, grader
- `server/simulation.py`: engagement and paid-lift simulation primitives
- `server/data_source.py`: synthetic brand/channel loading and seeding
- `models.py`: Pydantic action, observation, and state models
- `client.py`: typed client for interacting with the running server
- `inference.py`: baseline LLM runner
- `test_smoke.py`: end-to-end sanity checks

## Troubleshooting

### `budget_fractions` Validation Error In The Web UI

If you see a validation error for `budget_fractions`:

- make sure you are on Task 3 if you are sending budget values
- enter the value as a JSON-style list string, for example
  `[0.2, 0.2, 0.2, 0.2, 0.2]`
- provide exactly 5 values for Task 3
- make sure the values are non-negative and sum to about `1.0`

### Invalid `brand_id`

Make sure `brand_id` matches the task size:

- Task 1: `0`
- Task 2: `0-2`
- Task 3: `0-4`

### Inference Authentication Fails

Check:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

### Stale Local Database Behavior

Delete the SQLite file and reinitialize it:

```bash
rm -f data/social_media.db
uv run python init_sqlite_db.py
```

PowerShell:

```powershell
Remove-Item data/social_media.db -ErrorAction SilentlyContinue
uv run python init_sqlite_db.py
```

### Hugging Face Space Push Problems

- confirm the repo id is correct
- confirm you are authenticated with the required CLI tools
- make sure the Space exists or that you have permission to create it

## License

Apache-2.0
