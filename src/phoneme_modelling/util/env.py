import os
from pathlib import Path
from typing import Optional


def load_env(path: Optional[Path]) -> None:
    """KEY=value lines from a .env file into the environment, without overriding variables already set."""
    if path is None or not Path(path).exists():
        return
    for line in Path(path).read_text().splitlines():
        key, sep, value = line.partition("=")
        if sep and not line.startswith("#"):
            os.environ.setdefault(key.strip(), value.strip())
