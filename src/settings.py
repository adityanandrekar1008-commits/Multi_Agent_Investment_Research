"""Local settings: environment variables first, then the project's .env file.

.env is git-ignored, so personal values (such as the SEC EDGAR contact
email) stay out of the repository. Format: KEY=value, one per line.
"""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"


def _read_env_file(path=None):
    path = Path(path or ENV_FILE)
    values = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def get_setting(key, default=None):
    if os.environ.get(key):
        return os.environ[key]
    return _read_env_file().get(key, default)
