"""Cross-process runtime state shared between the web UI and the service.

The web UI (FastAPI) and the AlertWorker run in separate processes, so any
toggle that affects both must be persisted on disk. State lives as a tiny
flag file under ``log_dir`` (which both processes resolve from the same
config). File presence is the boolean — no parsing required.
"""

from __future__ import annotations

from pathlib import Path

_MUTE_FILENAME = "mute.flag"


def mute_flag_path(log_dir: str | Path) -> Path:
    """Return the absolute path of the mute flag file."""
    return Path(log_dir) / _MUTE_FILENAME


def is_muted(log_dir: str | Path) -> bool:
    """Return True if alert audio is currently muted."""
    return mute_flag_path(log_dir).exists()


def set_muted(log_dir: str | Path, muted: bool) -> bool:
    """Set the mute flag to *muted*. Returns the new state."""
    path = mute_flag_path(log_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    if muted:
        path.touch(exist_ok=True)
    else:
        path.unlink(missing_ok=True)
    return muted


def toggle_muted(log_dir: str | Path) -> bool:
    """Flip the mute flag. Returns the new state."""
    return set_muted(log_dir, not is_muted(log_dir))
