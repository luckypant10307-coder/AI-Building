"""Load a local .env into the environment, without a dependency.

Real environment variables win, so Render's dashboard settings always override
whatever is in a developer's .env. Lines are KEY=VALUE; # starts a comment.
"""

import os
from pathlib import Path


def load_dotenv(path=".env"):
    file = Path(path)
    if not file.exists():
        return 0
    loaded = 0
    for raw in file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded += 1
    return loaded
