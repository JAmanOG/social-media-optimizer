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

# Social Media Optimizer — OpenEnv RL Environment

A production-style reinforcement learning environment where an agent acts as a
social media strategist for a multi-brand digital agency.

The agent must balance engagement growth, delayed business outcomes, brand-level
coverage, and budget efficiency across tasks with increasing complexity.

## Why this environment

This project is designed for realistic policy learning instead of one-step
metric optimization. It includes:

- delayed conversion outcomes (not only immediate engagement)
- compliance and policy-risk penalties
- market trend regime shifts over an episode
- temporary platform shock events
- creative fatigue dynamics

Together these mechanics make the policy problem non-trivial, multi-objective,
and suitable for both LLM-agent baselines and RL training loops.

## Environment summary

| Component | Details |
|---|---|
| Action space | `brand_id` + `content_type` + `post_time_slot`; Task 3 also uses `budget_fractions` |
| Observation space | Per-brand recent performance, cadence, content history, follower profile, risk/fatigue state, episode counters |
| Reward | Dense reward composed of engagement, delayed conversions, diversification/coverage effects, and risk penalties |
| Episode lengths | Task 1: 7 steps, Task 2: 14 steps, Task 3: 21 steps |
| Data policy | Synthetic-only local data (safe to share) |

## Task design

### Task 1 — Single Brand Engagement (Easy)
Optimize one brand for short-horizon engagement quality.

Focus:
- timing and content choice
- cadence quality
- avoiding repeat-pattern fatigue

### Task 2 — Multi-Brand Content Mix (Medium)
Manage three brands with a balanced strategy.

Focus:
- preventing brand neglect
- balancing content distribution
- handling policy-risk side effects

### Task 3 — Budget Allocation + Cross-Brand (Hard)
Manage five channels and allocate daily paid budget while still making organic
decisions.

Focus:
- portfolio-level efficiency
- diversification under market shocks
- stable long-horizon reward under delayed outcomes

## Reward components (high level)

- immediate engagement quality
- delayed conversion realization
- portfolio paid-lift efficiency (Task 3)
- diversification / anti-neglect shaping
- policy/compliance and concentration penalties

Invalid actions consume a step and incur a penalty, so agents cannot exploit
stalling behavior.

## Data policy and safety

This repository is configured for synthetic-only runtime data.

- `data/social_media.db` is synthetic-seeded
- no CSV real-brand import path is used in normal flow
- smoke tests and setup scripts use synthetic-only defaults

## Quick start

### 1) Install

```bash
python3 -m venv venv
source venv/bin/activate
pip install uv
uv sync
```

### 2) Initialize synthetic local database

```bash
./venv/bin/python init_sqlite_db.py
```

### 3) Run smoke tests

```bash
SOCIAL_SQLITE_PATH=./data/social_media.db ./venv/bin/python test_smoke.py
```

### 4) Start API server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

## Running baseline inference

### Local Gemini (OpenAI-compatible endpoint)

```bash
LLM_PROVIDER=gemini \
GEMINI_MODEL=gemini-2.5-flash \
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/ \
GEMINI_API_KEY=<your_key> \
SOCIAL_SQLITE_PATH=./data/social_media.db \
./venv/bin/python inference.py
```

### Hugging Face Router

```bash
LLM_PROVIDER=huggingface \
HF_TOKEN=<your_token> \
HF_MODEL=Qwen/Qwen2.5-72B-Instruct \
HF_API_BASE_URL=https://router.huggingface.co/v1 \
SOCIAL_SQLITE_PATH=./data/social_media.db \
./venv/bin/python inference.py
```

## Docker

```bash
docker build -t social-media-optimizer .
docker run -p 7860:7860 social-media-optimizer
```

## Deploy to Hugging Face Space

```bash
openenv push <username>/social-media-optimizer
```

or use the helper script:

```bash
bash push_space.sh <username>/social-media-optimizer
```

## Project layout

```text
social-media-optimizer/
├── server/
│   ├── app.py                       # FastAPI/OpenEnv server wiring
│   ├── social_media_environment.py  # Core environment dynamics + reward/grader
│   ├── simulation.py                # Engagement simulation primitives
│   └── data_source.py               # Synthetic SQLite seeding + data access
├── models.py                        # Pydantic action/observation/state models
├── client.py                        # EnvClient integration
├── inference.py                     # LLM baseline runner
├── test_smoke.py                    # End-to-end smoke checks
├── init_sqlite_db.py                # Synthetic DB initializer
├── openenv.yaml                     # OpenEnv metadata
├── Dockerfile
├── pyproject.toml
└── README.md
```

## Troubleshooting

- `src refspec main does not match any`:
  create/switch to `main`, make a commit, then push.
- `Repository not found` on push:
  verify the remote URL and that the GitHub repo exists.
- inference fails with auth errors:
  confirm provider token and base URL are set correctly.
- stale local DB behavior:
  delete `data/social_media.db` and rerun `init_sqlite_db.py`.

## License

Apache-2.0
