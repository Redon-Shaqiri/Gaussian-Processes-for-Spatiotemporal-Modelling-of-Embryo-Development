import argparse
import time
from pathlib import Path

import numpy as np
from experiments.utils import save_json, step_done, summarise_trials

from gpembryos.logging_utils import info, init_logging
from gpembryos.utils import get_artifact_dir, update_latest_symlink

import gpytorch as gp
import pandas as pd
import scanpy as sc
import torch
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.feature_selection import mutual_info_regression
from gpembryos.evaluation import multioutput_metrics, get_multi_parameters
from gpembryos.models import MultitaskGPModel
from gpembryos.preprocessing import multiple_gene_expression_values, genes_variability_ranked
from gpembryos.training import variational_gp_training
from gpembryos.plotting import plot_training_history, plot_spatial_predictions, plot_predictions_vs_true, plot_spatial_predictions_uncertainty, plot_multi_calibration_curve

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
            "nu": args.nu,
            "learning_rate": args.learning_rate,
            "training_iterations": args.training_iterations,
            "early_stoppage_wait": args.early_stoppage_wait,
            "has_scheduler": args.has_scheduler,
            "batch_size": args.batch_size
        },
        "experiment": {
            "n_trials": args.n_trials,
            "test_size": args.test_size,
            "validation_size": args.validation_size,
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

    X_genes = data[:, genes].X
    if hasattr(X_genes, "toarray"):
        X_genes = X_genes.toarray()

    if gene is None:
        gene = genes[0]

    info(f"Target gene: {gene}")


    indices = np.arange(len(data))

    train_indices, test_indices = train_test_split(
        indices,
        test_size=params["experiment"]["test_size"],
        random_state=seed,
    )

    target_idx = genes.index(gene)

    y_target_train = X_genes[train_indices, target_idx]

    candidate_indices = [
        i for i in range(len(genes))
        if i != target_idx
    ]

    X_candidates_train = X_genes[
        np.ix_(train_indices, candidate_indices)
    ]

    mi = mutual_info_regression(
        X_candidates_train,
        y_target_train,
        random_state=seed,
    )


    ranking = np.argsort(mi)[::-1]

    ranked_indices = np.array(candidate_indices)[ranking]
    ranked_genes = np.array(genes)[ranked_indices]
    ranked_mi = mi[ranking]

    top_k = 4

    multi_output_genes = ([gene] + ranked_genes[:top_k].tolist())

    info(f"Selected multi-output genes: {multi_output_genes}")

    for selected_gene, score in zip(ranked_genes[:top_k], ranked_mi[:top_k]):
        info(f"MI({gene}, {selected_gene}) = {score:.4f}")


    x = np.asarray(data.obsm["spatial"], dtype=np.float64)

    y = np.log1p(multiple_gene_expression_values(data, multi_output_genes))

    info(f"Number of observations: {len(y)}")

    X_train = x[train_indices]
    X_test = x[test_indices]

    y_train = y[train_indices]
    y_test = y[test_indices]

    X_test_original = X_test.copy()

    X_train_mean = X_train.mean(axis=0)
    X_train_std = X_train.std(axis=0)

    X_train = ((X_train - X_train_mean) / X_train_std)

    X_test = ((X_test - X_train_mean)/ X_train_std)


    y_mean = y_train.mean(axis=0)
    y_std = y_train.std(axis=0)

    y_train = ((y_train - y_mean)/ y_std)

    y_test = ((y_test - y_mean)/ y_std)

    X_train = torch.tensor(X_train,dtype=torch.float64)

    X_test = torch.tensor(X_test, dtype=torch.float64)

    y_train = torch.tensor(y_train, dtype=torch.float64)

    y_test = torch.tensor(y_test, dtype=torch.float64)

    x_t, x_val, y_t, y_val = train_test_split(
        X_train,
        y_train,
        test_size=params["experiment"]["validation_size"],
        random_state=seed,
    )
    n_inducing = 500
    
    kmeans = KMeans(n_clusters=n_inducing, random_state=seed, n_init="auto")
    
    inducing_points = torch.tensor(kmeans.fit(x_t.cpu().numpy()).cluster_centers_, dtype=torch.float64)

    likelihood = gp.likelihoods.MultitaskGaussianLikelihood(num_tasks=top_k + 1).double()

    model = MultitaskGPModel(inducing_points=inducing_points, nu=params["model"]["nu"]).double()

    training_results = variational_gp_training(
        model=model,
        likelihood=likelihood,
        x_train_standardised=x_t,
        y_train_standardised=y_t,
        x_validation=x_val,
        y_validation=y_val,
        training_iterations=params["model"]["training_iterations"],
        learning_rate=params["model"]["learning_rate"],
        early_stoppage_wait=params["model"]["early_stoppage_wait"],
        has_scheduler=params["model"]["has_scheduler"],
        batch_size=params["model"]["batch_size"]
    )

    model = training_results["model"]
    likelihood = training_results["likelihood"]

    model_parameters = get_multi_parameters(model, likelihood)
    lengthscales = model_parameters["lengthscale"]

    with torch.no_grad(), gp.settings.fast_pred_var():
        predictions = likelihood(model(X_test))

    metrics = multioutput_metrics(y_test, predictions, multi_output_genes)

    elapsed = time.time() - t0

    spatial_data = pd.DataFrame({
        "X": X_test_original[:, 0],
        "Y": X_test_original[:, 1],
        "Z": X_test_original[:, 2],
        "true": y_test[:, 0].numpy(),
        "prediction": predictions.mean[:, 0].detach().numpy(),
        "uncertainty": predictions.stddev[:, 0].detach().numpy()
    })

    plot_training_history(training_results["training_losses"], training_results["validation_nlpds"], trial_dir / "training_history.png")

    plot_predictions_vs_true(y_test[:, 0].cpu().numpy(), predictions.mean[:, 0].detach().cpu().numpy(), trial_dir / "predictions_vs_true_multiMI.png")

    plot_spatial_predictions(spatial_data, trial_dir / "spatial_predictions_multiMI.png")

    plot_spatial_predictions_uncertainty(spatial_data, trial_dir / "spatial_predictions_uncertainty_multiMI.png", n_slices=5)

    calibration_results = plot_multi_calibration_curve(model=model, likelihood=likelihood, X_test=X_test, y_test=y_test[:, 0], output_idx=0, output_path=trial_dir / "calibration_curve_multiMI.png")

    metrics = {
        "trial": trial_idx,
        "seed": seed,
        "target_gene": gene,
        "multi_output_genes": multi_output_genes,
        "nu": params["model"]["nu"],
        "learning_rate": params["model"]["learning_rate"],
        "training_iterations": params["model"]["training_iterations"],
        "has_scheduler": params["model"]["has_scheduler"],
        "best_epoch": training_results["best_epoch"],
        "best_val_nlpd": training_results["best_validation_nlpd"],
        "elapsed": elapsed,
        "mace": calibration_results["mace"],
        "gene_metrics": multioutput_metrics(
            y_test,
            predictions,
            multi_output_genes
        ),
        **model_parameters,
    }
    if params["model"]["has_scheduler"]:
        metrics["learning_rates"] = training_results["learning_rates"]

    sox32_metrics = metrics["gene_metrics"][gene]

    info(f"SOX32 RMSE: {sox32_metrics['rmse']:.4f}")
    info(f"SOX32 MAE: {sox32_metrics['mae']:.4f}")
    info(f"SOX32 R²: {sox32_metrics['r2']:.4f}")
    info(f"SOX32 NLPD: {sox32_metrics['nlpd']:.4f}")
    info(f"SOX32 CRPS: {sox32_metrics['crps']:.4f}")
    info(f"Test SOX32 MACE: {calibration_results['mace']:.4f}")
    for latent_idx, ls in enumerate(lengthscales):
        info(
            f"Latent {latent_idx + 1} lengthscales: "
            f"x={ls[0]:.6f}, "
            f"y={ls[1]:.6f}, "
            f"z={ls[2]:.6f}"
        )

    info(f"Outputscales: {model_parameters['outputscale']}")

    info(f"Noise: {model_parameters['noise']}")
    
    save_json(trial_dir / "metrics.json", metrics)
    info(f"   Trial {trial_idx} done in {elapsed:.1f}s. Saved metrics.json")


if __name__ == "__main__":
    repo_root = Path(__file__).parent.parent
    parser = argparse.ArgumentParser(description=f"{EXPERIMENT}: Sparse Variational Gaussian Process")

    # Data
    parser.add_argument("--dataset", type=Path, required=True, help="Path to the .h5ad dataset") 
    parser.add_argument("--gene", type=str, default=None)
    # Model
    parser.add_argument("--nu", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--training-iterations", type=int, default=100)
    parser.add_argument("--early-stoppage-wait", type=int, default=20)
    parser.add_argument("--has-scheduler", type=bool, default=True)
    parser.add_argument("--batch-size", type=int, default=1000)
    # Experiment
    parser.add_argument("--n-trials",type=int,default=1,)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--validation-size", type=float, default=0.17)
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