"""Utilities shared across experiment scripts.

Importable from any script in `experiments/` as `from utils import ...`
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from gpembryos.logging_utils import info

# ============================================================
# JSON serialisation
# ============================================================


class SafeEncoder(json.JSONEncoder):
    """Serialises numpy scalars and arrays; raises on anything unexpected
    rather than silently coercing."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")


def save_json(path: Path, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, cls=SafeEncoder)


# ============================================================
# Completion checks (for --resume)
# ============================================================


def step_done(trial_dir: Path, sentinel: str) -> bool:
    """Check whether a named step has completed.

    Usage:
        if not step_done(trial_dir, "gp_predictions.npz"):
            ... run GP ...
        if not step_done(trial_dir, "metrics.json"):
            ... run evaluation ...
    """
    return (trial_dir / sentinel).exists()


# ============================================================
# Summary across trials
# ============================================================


def _flatten(obj: dict, prefix: str = "") -> dict:
    """Recursively flatten a nested dict to dot-separated keys.
    Keeps only numeric leaves."""
    items = {}
    for k, v in obj.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            items.update(_flatten(v, key))
        elif isinstance(v, (int, float)):
            items[key] = v
    return items


def summarise_trials(artifact_dir: Path, n_trials: int) -> None:
    rows = []
    for i in range(n_trials):
        mpath = artifact_dir / f"trial_{i}" / "metrics.json"
        if mpath.exists():
            with open(mpath) as f:
                rows.append(_flatten(json.load(f)))

    if not rows:
        return

    info(f"\n{'=' * 60}")
    info(f"Summary across {len(rows)} trials")
    info(f"{'=' * 60}")

    all_keys = sorted({k for r in rows for k in r})
    summary = {}
    for key in all_keys:
        vals = [r[key] for r in rows if key in r]
        summary[key] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }
        if len(vals) > 1:
            info(f"   {key}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
        else:
            info(f"   {key}: {vals[0]:.4f}")

    save_json(artifact_dir / "summary.json", summary)
    info(f"Saved: summary.json")
