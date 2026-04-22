import os
from pathlib import Path


def _parse_env_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in stripped:
        return None

    key, value = stripped.split("=", 1)
    key = key.strip()
    value = value.strip()

    if not key:
        return None

    if value and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]

    return key, value


def load_project_env(start_file: str | None = None) -> None:
    """
    Loads .env files from the current script directory and its parent project root.
    Existing environment variables are preserved.
    """
    start_path = Path(start_file).resolve() if start_file else Path(__file__).resolve()
    script_dir = start_path.parent
    # env_utils.py is always at src/env_utils.py, so its parent's parent is the project root
    # regardless of how deep the calling script is nested under src/
    project_root = Path(__file__).resolve().parent.parent

    seen: set[Path] = set()
    candidates = [project_root / ".env", script_dir / ".env"]
    for env_path in candidates:
        if env_path in seen:
            continue
        seen.add(env_path)
        if not env_path.exists():
            continue
        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                parsed = _parse_env_line(line)
                if not parsed:
                    continue
                key, value = parsed
                os.environ.setdefault(key, value)
        except OSError:
            continue
