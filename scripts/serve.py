"""Start the FastAPI inference service.

Usage:
    python scripts/serve.py --config configs/app.yaml
Then open http://localhost:8000/docs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ffsd.core.config import load_app_config  # noqa: E402
from ffsd.service.api import create_app  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve detection API")
    parser.add_argument("--config", default="configs/app.yaml")
    args = parser.parse_args()

    cfg = load_app_config(args.config)
    app = create_app(args.config)
    uvicorn.run(app, host=cfg.server.host, port=cfg.server.port)


if __name__ == "__main__":
    main()
