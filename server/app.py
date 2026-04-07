"""FastAPI application for the Social Media Optimizer Environment."""

import os

from openenv.core.env_server import create_app

# Support both in-repo and standalone imports
try:
    from ..models import SocialAction, SocialObservation
    from .social_media_environment import SocialMediaOptimizerEnv
except ImportError as e:
    if "relative import" not in str(e) and "no known parent package" not in str(e):
        raise
    from models import SocialAction, SocialObservation
    from server.social_media_environment import SocialMediaOptimizerEnv

# OpenEnv expects ENABLE_WEB_INTERFACE, not OPENENV_ENABLE_WEB_INTERFACE.
# Default to enabling the UI so the /web endpoint is available locally and in Spaces.
os.environ.setdefault("ENABLE_WEB_INTERFACE", "true")

# Create the FastAPI app
app = create_app(
    SocialMediaOptimizerEnv,
    SocialAction,
    SocialObservation,
    env_name="social_media_optimizer",
    max_concurrent_envs=4,
)


def main():
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
