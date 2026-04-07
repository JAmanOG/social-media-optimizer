---
title: Social Media Optimizer
emoji: 📱
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

An RL environment for training agents to act as social media strategists for a
multi-brand digital marketing agency.

## 🎯 What It Does

The agent manages a portfolio of brands and must learn to optimize:

- posting time
- content type
- paid budget allocation

The environment is designed around real agency workflow themes from the
hackathon brief: dynamic optimization, multi-brand coordination, and
portfolio-level tradeoffs instead of static reporting.

When `../test_data` is available, the env bootstraps a local SQLite database
from those CSVs and then loads channel-aware brand profiles from SQLite before
falling back to fully synthetic generation.

## 🧩 Environment Description

| Component | Details |
|---|---|
| **Action space** | Brand selection (int), content type (reel/carousel/static), time slot (0-5), daily budget fractions (Task 3) |
| **Observation space** | Per-brand engagement history, average engagement, recent content/slot usage, cadence, follower count, budget state, day/step counters |
| **Reward** | Dense per-step reward. Task 2 penalizes neglect/imbalance. Task 3 rewards portfolio efficiency and diversification. |
| **Episode length** | Task-dependent: 7 / 14 / 21 steps |

## 📋 Tasks

### Task 1 — Single Brand Engagement (Easy)
Maximise engagement for a single brand over 7 simulated days.

### Task 2 — Multi-Brand Content Mix (Medium)
Manage 3 brands simultaneously, optimising content type ratio and
posting schedule for each. Neglecting brands or over-focusing on one brand
reduces the score.

### Task 3 — Budget Allocation + Cross-Brand (Hard)
Allocate a shared daily budget across 5 brands while choosing organic
post timing and content type. The budget is reallocated each day, and the
grader rewards portfolio efficiency plus diversification.

## 🚀 Quick Start

### Install & Run Locally

```bash
# Create virtual environment
python3 -m venv venv && source venv/bin/activate

# Install dependencies
pip install uv
uv sync

# Start the server
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Local LLM Testing With Gemini

The inference script defaults to Gemini for local testing via Google's
OpenAI-compatible endpoint.

```bash
cp .env.example .env
# fill in GEMINI_API_KEY
./venv/bin/python inference.py
```

### Using SQLite As the Local Database

This is the recommended local setup right now. The database is seeded from
`../test_data`, so the CSVs remain your bootstrap source while the environment
reads through SQLite.

```bash
SOCIAL_DATA_ROOT=../test_data ./venv/bin/python init_sqlite_db.py
SOCIAL_SQLITE_PATH=./data/social_media.db ./venv/bin/python test_smoke.py
SOCIAL_SQLITE_PATH=./data/social_media.db ./venv/bin/python inference.py
```

With the current dataset, the database stores normalized raw posts plus derived
channel profiles. Task 3 naturally uses the five channels:
Instagram, Facebook, LinkedIn, YouTube, and X.

Key local variables:

```bash
LLM_PROVIDER=gemini
GEMINI_MODEL=gemini-2.5-flash
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
```

### Production / Hugging Face Router

```bash
LLM_PROVIDER=huggingface \
HF_TOKEN=... \
HF_MODEL=Qwen/Qwen2.5-72B-Instruct \
HF_API_BASE_URL=https://router.huggingface.co/v1 \
./venv/bin/python inference.py
```

### Docker

```bash
docker build -t social-media-optimizer .
docker run -p 7860:7860 social-media-optimizer
```

## 🏗️ Project Structure

```
social-media-optimizer/
├── models.py                 # Pydantic Action/Observation/State types
├── client.py                 # EnvClient subclass
├── inference.py              # Baseline LLM inference script (Gemini local, HF prod)
├── init_sqlite_db.py         # Seed SQLite database from test_data CSVs
├── test_smoke.py             # Comprehensive smoke test suite
├── openenv.yaml              # OpenEnv metadata
├── pyproject.toml            # Python package config
├── Dockerfile                # Container build (at root)
├── .env.example              # Environment variable template
├── data/                     # SQLite database (auto-generated)
│   └── social_media.db
├── server/
│   ├── app.py                # FastAPI server
│   ├── social_media_environment.py  # Core RL environment
│   ├── simulation.py         # Engagement model + brand generator
│   └── data_source.py        # SQLite + CSV data pipeline
└── README.md
```

## Reward Design

- **Task 1:** Direct engagement reward with posting-cadence and fatigue effects.
- **Task 2:** Engagement reward plus brand-coverage bonus, with penalties for neglect and over-concentration.
- **Task 3:** Organic engagement plus portfolio paid lift, with penalties for over-concentrated budget allocations.

## Notes

- Invalid actions count as wasted turns, so agents cannot stall episodes indefinitely.
- Task 3 models the budget as a fresh daily decision, not a one-time episode budget.
- The smoke test includes checks for the old exploit paths: single-brand neglect and all-budget-to-one-brand.

## 🔗 Deploy to HuggingFace

```bash
openenv push <username>/social-media-optimizer
```
