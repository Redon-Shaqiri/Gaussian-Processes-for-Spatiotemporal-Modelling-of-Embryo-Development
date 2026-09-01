import argparse
import time
from pathlib import Path

import numpy as np
from experiments.utils import save_json, step_done, summarise_trials

from gpembryos.logging_utils import info, init_logging
from gpembryos.utils import get_artifact_dir, update_latest_symlink

import pandas as pd
import scanpy as sc
import torch
import jax
import jax.numpy as jnp
import gpjax as gpx
from gpjax.likelihoods import (HeteroscedasticGaussian, LogNormalTransform)
from gpjax.variational_families import (HeteroscedasticVariationalFamily, VariationalGaussianInit)
import optax as ox
import matplotlib.pyplot as plt
import properscoring as ps
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from gpembryos.preprocessing import gene_expression_values, genes_variability_ranked
from gpembryos.plotting import plot_spatial_predictions_aleatoric_uncertainty, plot_spatial_predictions, plot_predictions_vs_true, plot_spatial_predictions_uncertainty, plot_calibration_curve_het

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
            "dataset": str(args.dataset),
            "gene": args.gene,
        },
        "model": {
            "learning_rate": args.learning_rate,
            "training_iterations": args.training_iterations,
            "batch_size": args.batch_size,
        },
        "experiment": {
            "n_trials": args.n_trials,
            "test_size": args.test_size,
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

    dataset_path = params["data"]["dataset"]
    gene = params["data"]["gene"]

    info(f"Loading dataset: {dataset_path}")
    info(f"Gene: {gene}")

    data = sc.read_h5ad(dataset_path)

    genes = genes_variability_ranked(data, n_top_genes=50)
    gene = params["data"]["gene"]

    if gene is None:
        gene = genes[0]

    info(f"Gene: {gene}")

    x =  np.asarray(data.obsm["spatial"], dtype=np.float64)
    y = np.log1p(gene_expression_values(data, gene))
    info(f"Number of observations: {len(y)}")

    jax.config.update("jax_enable_x64", True)

    X_train, X_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=params["experiment"]["test_size"],
        random_state=seed,
    )


    X_test_original = X_test
    X_train_mean = X_train.mean(axis=0)
    X_train_std = X_train.std(axis=0)

    X_train = (X_train - X_train_mean) / X_train_std
    X_test = (X_test - X_train_mean) / X_train_std

    y_mean = y_train.mean()
    y_std = y_train.std()

    y_train = (y_train - y_mean) / y_std
    y_test = (y_test - y_mean) / y_std


    X_train_jax = jnp.asarray(X_train, dtype=jnp.float64)
    y_train_jax = jnp.asarray(y_train, dtype=jnp.float64).reshape(-1, 1)


    X_test_jax = jnp.asarray(X_test, dtype=jnp.float64)
    y_test_jax = jnp.asarray(y_test, dtype=jnp.float64).reshape(-1, 1)

    train = gpx.Dataset(X=X_train_jax, y=y_train_jax)

    M = 1000

    kmeans = KMeans(n_clusters=M, random_state=seed, n_init="auto")
    kmeans.fit(X_train)
    # Initialise the inducing points for both the signal and noise with the same points
    z_signal = jnp.asarray(kmeans.cluster_centers_, dtype=jnp.float64)
    z_noise = z_signal.copy()


    print("Training data:", X_train_jax.shape)
    print("Signal inducing points:", z_signal.shape)
    print("Noise inducing points:", z_noise.shape)
    # Initialise
    signal_kernel = gpx.kernels.Matern12(active_dims=[0, 1, 2], lengthscale=jnp.array([0.2, 0.2, 0.2], dtype=jnp.float64), variance=1.0)
    signal_prior = gpx.gps.Prior(mean_function=gpx.mean_functions.Constant(constant=jnp.mean(y_train_jax)), kernel=signal_kernel)

    # Matern12, Matern32, Matern52
    noise_kernel = gpx.kernels.Matern52(active_dims=[0, 1, 2], lengthscale=jnp.array([0.5, 0.5, 0.5], dtype=jnp.float64),variance=0.1)
    noise_prior = gpx.gps.Prior(mean_function=gpx.mean_functions.Constant(constant=-2.0), kernel=noise_kernel)

    likelihood = HeteroscedasticGaussian(
        num_datapoints=train.n,
        noise_prior=noise_prior,
        noise_transform=LogNormalTransform(),
    )

    posterior = signal_prior * likelihood

    q_sparse = HeteroscedasticVariationalFamily(
        posterior=posterior,
        signal_init=VariationalGaussianInit(inducing_inputs=z_signal),
        noise_init=VariationalGaussianInit(inducing_inputs=z_noise),
    )

    optimiser = ox.adam(
        params["model"]["learning_rate"]
    )

    q_sparse_trained, history = gpx.fit(
        model=q_sparse,
        objective=lambda m, d: -gpx.objectives.heteroscedastic_elbo(m, d),
        train_data=train,
        optim=optimiser,
        num_iters=params["model"]["training_iterations"],
        batch_size=params["model"]["batch_size"],
        key=jax.random.PRNGKey(seed),
        verbose=False,
    )


    losses = np.asarray(history)

    plt.figure(figsize=(8, 5))
    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("Negative ELBO")
    plt.title("Heteroscedastic GP Training Loss")
    plt.tight_layout()
    plt.savefig(trial_dir / "training_loss.png", dpi=300)
    plt.close()


    signal_pred, noise_pred = q_sparse_trained.predict_latents(X_test_jax)


    mf = np.asarray(signal_pred.mean).squeeze()
    vf = np.asarray(signal_pred.variance).squeeze()

    mg = np.asarray(noise_pred.mean).squeeze()
    vg = np.asarray(noise_pred.variance).squeeze()

    # Compute aleatoric uncertainty
    aleatoric_variance = np.exp(mg + 0.5 * vg)
    aleatoric_std = np.sqrt(aleatoric_variance)

    predictive = likelihood.predict(signal_pred, noise_pred)
    predictions_mean = np.asarray(predictive.mean).squeeze()
    predictions_var = np.asarray(predictive.variance).squeeze()
    predictions_std = np.sqrt(predictions_var)

    y_test = np.asarray(y_test).reshape(-1)
    # Compute metrics
    rmse = np.sqrt(np.mean((predictions_mean - y_test) ** 2))
    mae = np.mean(np.abs(predictions_mean - y_test))
    r2 = r2_score(y_test, predictions_mean)
    nlpd = (0.5*np.log(2*np.pi*predictions_var) + (y_test - predictions_mean)**2 / (2*predictions_var)).mean()
    crps = ps.crps_gaussian(y_test, mu = predictions_mean, sig = predictions_std)
    crps_mean = np.mean(crps)    
    # Retrieve learned parameters
    trained_signal_kernel = (q_sparse_trained.signal_variational.posterior.prior.kernel)
    trained_signal_mean = (q_sparse_trained.signal_variational.posterior.prior.mean_function)
    trained_noise_kernel = (q_sparse_trained.noise_variational.posterior.prior.kernel)
    trained_noise_mean = (q_sparse_trained.noise_variational.posterior.prior.mean_function)
    signal_lengthscales = np.asarray(trained_signal_kernel.lengthscale.unwrap())
    signal_variance = float(trained_signal_kernel.variance.unwrap())
    signal_mean = float(trained_signal_mean.constant)
    noise_lengthscales = np.asarray(trained_noise_kernel.lengthscale.unwrap())
    noise_variance = float(trained_noise_kernel.variance.unwrap())
    noise_mean = float(trained_noise_mean.constant)
    
    
    model_parameters = {
        "signal_lengthscale": np.round(signal_lengthscales.tolist(), 3),
        "signal_variance": np.round(signal_variance, 3),
        "signal_mean": np.round(signal_mean, 3),
        "noise_lengthscale": np.round(noise_lengthscales.tolist(), 3),
        "noise_variance": np.round(noise_variance, 3),
        "noise_mean": np.round(noise_mean, 3)
    }

    elapsed = time.time() - t0

    # Plot aleatoric uncertainty
    aleatoric = pd.DataFrame({
        "X": X_test_original[:, 0],
        "Y": X_test_original[:, 1],
        "Z": X_test_original[:, 2],
        "prediction": predictions_mean,
        "uncertainty": aleatoric_std,
        })

    spatial_data = pd.DataFrame({
        "X": X_test_original[:, 0],
        "Y": X_test_original[:, 1],
        "Z": X_test_original[:, 2],
        "true": y_test,
        "prediction": predictions_mean,
        "uncertainty": predictions_std
        })

    plot_spatial_predictions_aleatoric_uncertainty(aleatoric, trial_dir / "spatial_predictions_aleatoric_uncertainty_het.png", n_slices=5)

    plot_predictions_vs_true(y_test, predictions_mean, trial_dir / "predictions_vs_true_het.png")

    plot_spatial_predictions(spatial_data, trial_dir / "spatial_predictions_het.png")

    plot_spatial_predictions_uncertainty(spatial_data, trial_dir / "spatial_predictions_uncertainty_het.png", n_slices=5)
    
    calibration_results = plot_calibration_curve_het(predictive= predictive, y_test=y_test, output_path=trial_dir / "calibration_curve_het.png")

    metrics = {
        "trial": trial_idx,
        "seed": seed,
        "learning_rate": params["model"]["learning_rate"],
        "training_iterations": params["model"]["training_iterations"],
        "batch_size": params["model"]["batch_size"],
        "rmse": np.round(rmse, 3),
        "mae": np.round(mae, 3),
        "r2": np.round(r2, 3),
        "nlpd": np.round(nlpd, 3),
        "crps": np.round(crps_mean, 3),
        "mace": np.round(calibration_results['mace'], 3),
        **model_parameters
    }    
    

    info(f"Test RMSE: {metrics['rmse']:.4f}")
    info(f"Test MAE: {metrics['mae']:.4f}")
    info(f"Test R²: {metrics['r2']:.4f}")
    info(f"Test NLPD: {metrics['nlpd']:.4f}")
    info(f"Test CRPS: {metrics['crps']:.4f}")
    info(f"Test MACE: {calibration_results['mace']:.4f}")
    save_json(trial_dir / "metrics.json", metrics)
    info(f"   Trial {trial_idx} done in {elapsed:.1f}s. Saved metrics.json")
  


if __name__ == "__main__":
    repo_root = Path(__file__).parent.parent
    parser = argparse.ArgumentParser(description=f"{EXPERIMENT}: Sparse Variational Gaussian Process")

    # Data
    parser.add_argument("--dataset", type=Path, required=True, help="Path to the .h5ad dataset") 
    parser.add_argument("--gene", type=str, default=None)
    # Model
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--training-iterations", type=int, default=1250)
    parser.add_argument("--batch-size", type=int, default=1000)
    # Experiment
    parser.add_argument("--n-trials",type=int,default=1,)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
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