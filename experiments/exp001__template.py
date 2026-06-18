import argparse
import time
from pathlib import Path

import numpy as np
from utils import save_json, step_done, summarise_trials

from gpembryos.logging_utils import info, init_logging
from gpembryos.utils import get_artifact_dir, update_latest_symlink

EXPERIMENT = Path(__file__).stem.split("__")[0]


# ============================================================
# Params
# ============================================================


def load_or_create_params(artifact_dir, is_resume, args):
    params_file = artifact_dir / "params.json"
    if is_resume and params_file.exists():
        import json

        with open(params_file) as f:
            params = json.load(f)
        info(f"Loaded params from {params_file}")
        return params

    params = {
        "data": {
            "some_path": str(args.some_path),
            "some_int": args.some_int,
        },
        "model": {
            "some_param": args.some_param,
        },
        "experiment": {
            "n_trials": args.n_trials,
            "base_seed": args.seed,
        },
        "device": args.device,
    }
    save_json(params_file, params)
    info(f"Saved params to {params_file}")
    return params


# ============================================================
# Core logic
# ============================================================


def run_trial(trial_idx, trial_dir, params, seed):
    if step_done(trial_dir, "metrics.json"):
        info(f"   Trial {trial_idx} already complete, skipping.")
        return

    trial_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(seed)
    t0 = time.time()

    # placeholder
    if not step_done(trial_dir, "step1_output.npz"):
        intermediate = {"placeholder": np.float32(1.0)}
        np.savez(trial_dir / "step1_output.npz", **intermediate)
        info(f"   Saved: step1_output.npz")
    else:
        info(f"   Loading cached step1_output.npz")
        intermediate = dict(np.load(trial_dir / "step1_output.npz"))

    results = {"some_metric": 42.0}

    elapsed = time.time() - t0
    metrics = {
        "trial": trial_idx,
        "seed": seed,
        "elapsed": elapsed,
        **results,
    }
    save_json(trial_dir / "metrics.json", metrics)
    info(f"   Trial {trial_idx} done in {elapsed:.1f}s. Saved metrics.json")


if __name__ == "__main__":
    repo_root = Path(__file__).parent.parent
    parser = argparse.ArgumentParser(description=f"{EXPERIMENT}: example template")

    # Data
    parser.add_argument("--some-path", type=Path, default=repo_root / "data" / "something")
    parser.add_argument("--some-int", type=int, default=100)
    # Model
    parser.add_argument("--some-param", type=float, default=0.5)
    # Experiment
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-trials", type=int, default=1)
    parser.add_argument("--resume", nargs="?", const="latest", default=None)

    args = parser.parse_args()

    artifact_dir, is_resume = get_artifact_dir(repo_root, EXPERIMENT, args.resume)
    update_latest_symlink(artifact_dir)
    init_logging(level="debug", filename=str(artifact_dir / f"{EXPERIMENT}.log"))

    params = load_or_create_params(artifact_dir, is_resume, args)
    base_seed = params["experiment"]["base_seed"]
    n_trials = params["experiment"]["n_trials"]

    info("=" * 60)
    info(f"{EXPERIMENT}  [{n_trials} trial{'s' if n_trials > 1 else ''}]")
    info("=" * 60)
    info(f"Artifact dir: {artifact_dir}")

    for trial_idx in range(n_trials):
        trial_seed = base_seed + trial_idx
        trial_dir = artifact_dir / f"trial_{trial_idx}"
        info(f"\n{'=' * 60}")
        info(f"Trial {trial_idx}/{n_trials}  (seed={trial_seed})")
        info(f"{'=' * 60}")
        run_trial(trial_idx, trial_dir, params, trial_seed)

    if n_trials > 1:
        summarise_trials(artifact_dir, n_trials)

    info("\n" + "=" * 60)
    info("DONE")
    info(f"Artifacts: {artifact_dir}")
