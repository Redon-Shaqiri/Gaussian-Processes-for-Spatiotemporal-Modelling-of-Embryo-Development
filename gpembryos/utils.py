"""Shared utilities: artifact directory management and `latest` symlink."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

TIMESTAMP_FMT = "%Y-%m-%d_%H%M%S"


def get_artifact_dir(
    repo_root: Path,
    experiment: str,
    resume: str | None,
) -> tuple[Path, bool]:
    """Resolve the artifact directory for an experiment.

    Args:
        repo_root: project root (the dir containing `artifacts/`).
        experiment: e.g. "exp001".
        resume:
            - None: create a fresh timestamped dir.
            - "latest": resolve via `artifacts/<exp>/latest` symlink.
            - "<timestamp>": use `artifacts/<exp>/<timestamp>/`.

    Returns:
        (artifact_dir, is_resume)
    """
    base = Path(repo_root) / "artifacts" / experiment
    base.mkdir(parents=True, exist_ok=True)

    if resume is None:
        ts = datetime.now().strftime(TIMESTAMP_FMT)
        artifact_dir = base / ts
        artifact_dir.mkdir(parents=True, exist_ok=False)
        return artifact_dir, False

    if resume == "latest":
        latest = base / "latest"
        if not latest.exists():
            raise FileNotFoundError(f"No 'latest' symlink at {latest}; nothing to resume.")
        artifact_dir = latest.resolve()
    else:
        artifact_dir = base / resume
        if not artifact_dir.exists():
            raise FileNotFoundError(f"Run dir {artifact_dir} does not exist.")

    return artifact_dir, True


def update_latest_symlink(artifact_dir: Path) -> None:
    """Point `<exp>/latest` at `artifact_dir`. Replaces any existing symlink."""
    artifact_dir = Path(artifact_dir)
    latest = artifact_dir.parent / "latest"
    # Use a relative target so the symlink survives directory moves.
    target = artifact_dir.name
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    os.symlink(target, latest)
