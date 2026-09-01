import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import norm

def plot_training_history(train_losses, val_nlpds, output_path,):
    plt.figure()

    plt.plot(train_losses, label="Training")
    plt.plot(val_nlpds, label="Validation")

    plt.xlabel("Iteration")
    plt.ylabel("Loss / NLPD")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_predictions_vs_true(y_true, y_pred, output_path):
    fig, ax = plt.subplots()

    ax.scatter(y_true, y_pred, s=10)
    limits = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max()),]
    ax.plot(limits, limits, linestyle="--")

    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_spatial_predictions(spatial_data, output_path, n_slices=5):
    z_values = np.linspace(spatial_data["Z"].min(), spatial_data["Z"].max(), n_slices)

    fig, axes = plt.subplots(n_slices, 2, figsize=(10, 4 * n_slices))

    if n_slices == 1:
        axes = np.expand_dims(axes, axis=0)


    vmin = min(spatial_data["true"].min(), spatial_data["prediction"].min())
    vmax = max(spatial_data["true"].max(), spatial_data["prediction"].max())

    for i, z0 in enumerate(z_values):

        slice_data = spatial_data[np.abs(spatial_data["Z"] - z0) < 7]

        sc1 = axes[i, 0].scatter(
            slice_data["X"],
            slice_data["Y"],
            c=slice_data["prediction"],
            s=20,
            vmin=vmin,
            vmax=vmax
        )

        axes[i, 0].set_title(f"Predicted expression, Z ≈ {z0:.0f}")
        axes[i, 0].set_xlabel("X")
        axes[i, 0].set_ylabel("Y")
        axes[i, 0].set_aspect("equal")

        sc2 = axes[i, 1].scatter(
            slice_data["X"],
            slice_data["Y"],
            c=slice_data["true"],
            s=20,
            vmin=vmin,
            vmax=vmax
        )

        axes[i, 1].set_title(f"True expression, Z ≈ {z0:.0f}")
        axes[i, 1].set_xlabel("X")
        axes[i, 1].set_ylabel("Y")
        axes[i, 1].set_aspect("equal")

    fig.subplots_adjust(right=0.88, hspace=0.35, wspace=0.25)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.70])
    fig.colorbar(sc2, cax=cbar_ax, label="Gene expression")

    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.close(fig)




def plot_spatial_predictions_uncertainty(spatial_data, output_path, n_slices=5):
    z_values = np.linspace(spatial_data["Z"].min(), spatial_data["Z"].max(), n_slices)
    fig, axes = plt.subplots(n_slices, 2, figsize=(10, 4 * n_slices))

    if n_slices == 1:
        axes = np.expand_dims(axes, axis=0)

    pred_vmin = spatial_data["prediction"].min()
    pred_vmax = spatial_data["prediction"].max()

    uncertainty_vmin = spatial_data["uncertainty"].min()
    uncertainty_vmax = spatial_data["uncertainty"].max()

    for i, z0 in enumerate(z_values):

        slice_data = spatial_data[np.abs(spatial_data["Z"] - z0) < 7]


        sc1 = axes[i, 0].scatter(slice_data["X"], slice_data["Y"], c=slice_data["prediction"], s=20, vmin=pred_vmin, vmax=pred_vmax)

        axes[i, 0].set_title(f"Predicted expression, Z ≈ {z0:.0f}")
        axes[i, 0].set_xlabel("X")
        axes[i, 0].set_ylabel("Y")
        axes[i, 0].set_aspect("equal")

        sc2 = axes[i, 1].scatter(slice_data["X"], slice_data["Y"], c=slice_data["uncertainty"], s=20, vmin=uncertainty_vmin, vmax=uncertainty_vmax)

        axes[i, 1].set_title(f"Predictive uncertainty, Z ≈ {z0:.0f}")
        axes[i, 1].set_xlabel("X")
        axes[i, 1].set_ylabel("Y")
        axes[i, 1].set_aspect("equal")

    fig.subplots_adjust(right=0.88, hspace=0.35, wspace=0.25)
    cbar_ax1 = fig.add_axes([0.90, 0.53, 0.02, 0.35])
    fig.colorbar(sc1, cax=cbar_ax1, label="Predicted expression")
    cbar_ax2 = fig.add_axes([0.90, 0.12, 0.02, 0.35])
    fig.colorbar(sc2, cax=cbar_ax2, label="Predictive standard deviation")

    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.close(fig)




def plot_spatial_predictions_aleatoric_uncertainty(spatial_data, output_path, n_slices=5):
    z_values = np.linspace(spatial_data["Z"].min(), spatial_data["Z"].max(), n_slices)

    fig, axes = plt.subplots(n_slices, 2, figsize=(10, 4 * n_slices))

    if n_slices == 1:
        axes = np.expand_dims(axes, axis=0)

    pred_vmin = spatial_data["prediction"].min()
    pred_vmax = spatial_data["prediction"].max()

    uncertainty_vmin = spatial_data["uncertainty"].min()
    uncertainty_vmax = spatial_data["uncertainty"].max()

    for i, z0 in enumerate(z_values):

        slice_data = spatial_data[np.abs(spatial_data["Z"] - z0) < 7]


        sc1 = axes[i, 0].scatter(slice_data["X"], slice_data["Y"], c=slice_data["prediction"], s=20, vmin=pred_vmin, vmax=pred_vmax)

        axes[i, 0].set_title(f"Predicted expression, Z ≈ {z0:.0f}")
        axes[i, 0].set_xlabel("X")
        axes[i, 0].set_ylabel("Y")
        axes[i, 0].set_aspect("equal")

        sc2 = axes[i, 1].scatter(slice_data["X"], slice_data["Y"], c=slice_data["uncertainty"], s=20, vmin=uncertainty_vmin, vmax=uncertainty_vmax)

        axes[i, 1].set_title(f"Aleatoric uncertainty, Z ≈ {z0:.0f}")
        axes[i, 1].set_xlabel("X")
        axes[i, 1].set_ylabel("Y")
        axes[i, 1].set_aspect("equal")

    fig.subplots_adjust(right=0.88, hspace=0.35, wspace=0.25)
    cbar_ax1 = fig.add_axes([0.90, 0.53, 0.02, 0.35])
    fig.colorbar(sc1, cax=cbar_ax1, label="Predicted expression")
    cbar_ax2 = fig.add_axes([0.90, 0.12, 0.02, 0.35])
    fig.colorbar(sc2, cax=cbar_ax2, label="Aleatoric standard deviation")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.close(fig)



def plot_calibration_curve(model, likelihood, X_test, y_test, output_path, n_levels=9):
    model.eval()
    likelihood.eval()
    y_test = y_test.squeeze()

    with torch.no_grad():
        latent_dist = model(X_test)
        predictive_dist = likelihood(latent_dist)
        mean = predictive_dist.mean
        variance = predictive_dist.variance

    std = torch.sqrt(torch.clamp(variance, min=1e-12))
    levels = np.linspace(0.1, 0.9, n_levels)
    empirical_coverage = []

    for p in levels:
        a = norm.ppf((p + 1) / 2)
        lower = mean - a * std
        upper = mean + a * std
        inside = ((y_test >= lower) & (y_test <= upper))
        coverage = inside.double().mean().item()
        empirical_coverage.append(coverage)

    empirical_coverage = np.array(empirical_coverage)
    mace = np.mean(np.abs(empirical_coverage - levels))
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    ax.plot(levels, empirical_coverage, marker="o", label="GP")
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Predictive Uncertainty Calibration")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.legend()
    ax.grid(alpha=0.3)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.close(fig)

    return {
        "levels": levels,
        "empirical_coverage": empirical_coverage,
        "mace": mace
    }

def plot_regime_class(model, spatial_data, X_test, output_path, n_slices=5):
    kernel = model.covar_module
    center = kernel.boundary_center.detach()
    radius = torch.nn.functional.softplus(kernel.boundary_radius_g).detach()
    width = torch.exp(kernel.log_width).detach()
    r = torch.sqrt(torch.sum((X_test - center) ** 2, dim=-1))
    gate = torch.sigmoid((r - radius)/width)
    gate = gate.detach().cpu().numpy()
    spatial_data = spatial_data.copy()

    spatial_data["gate"] = gate

    spatial_data["regime"] = np.where(spatial_data["gate"] < 0.5, 1, 2)

    z_values = np.linspace(spatial_data["Z"].min(), spatial_data["Z"].max(), n_slices)

    fig, axes = plt.subplots(
        n_slices,
        2,
        figsize=(10, 4 * n_slices)
    )

    if n_slices == 1:
        axes = np.expand_dims(axes, axis=0)


    pred_vmin = spatial_data["prediction"].min()
    pred_vmax = spatial_data["prediction"].max()


    for i, z0 in enumerate(z_values):

        slice_data = spatial_data[np.abs(spatial_data["Z"] - z0) < 7]

        sc1 = axes[i, 0].scatter(
            slice_data["X"],
            slice_data["Y"],
            c=slice_data["prediction"],
            s=20,
            vmin=pred_vmin,
            vmax=pred_vmax
        )

        axes[i, 0].set_title(
            f"Predicted expression, Z ≈ {z0:.0f}"
        )

        axes[i, 0].set_xlabel("X")
        axes[i, 0].set_ylabel("Y")
        axes[i, 0].set_aspect("equal")


        sc2 = axes[i, 1].scatter(
            slice_data["X"],
            slice_data["Y"],
            c=slice_data["regime"],
            s=20,
            vmin=1,
            vmax=2
        )

        axes[i, 1].set_title(
            f"Changepoint regime, Z ≈ {z0:.0f}"
        )

        axes[i, 1].set_xlabel("X")
        axes[i, 1].set_ylabel("Y")
        axes[i, 1].set_aspect("equal")


    fig.subplots_adjust(right=0.88, hspace=0.35, wspace=0.25)
    cbar_ax1 = fig.add_axes([0.90, 0.53, 0.02, 0.35])
    fig.colorbar(sc1, cax=cbar_ax1, label="Predicted expression")
    cbar_ax2 = fig.add_axes([0.90, 0.12, 0.02, 0.35])
    cbar2 = fig.colorbar(sc2, cax=cbar_ax2, label="Regime")
    cbar2.set_ticks([1, 2])
    cbar2.set_ticklabels(["Regime 1", "Regime 2"])

    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.close(fig)


def plot_multi_calibration_curve(model, likelihood, X_test, y_test, output_path, n_levels=9,output_idx=0):
    model.eval()
    likelihood.eval()
    y_test = y_test.reshape(-1)
    with torch.no_grad():
        latent_dist = model(X_test)
        predictive_dist = likelihood(latent_dist)
        mean = predictive_dist.mean[..., output_idx].reshape(-1)
        variance = predictive_dist.variance[..., output_idx].reshape(-1)

    std = torch.sqrt(torch.clamp(variance, min=1e-12))
    levels = np.linspace(0.1, 0.9, n_levels)
    empirical_coverage = []

    for p in levels:
        a = norm.ppf((p + 1) / 2)
        lower = mean - a * std
        upper = mean + a * std
        inside = ((y_test >= lower) & (y_test <= upper))
        coverage = inside.double().mean().item()
        empirical_coverage.append(coverage)

    empirical_coverage = np.array(empirical_coverage)
    mace = np.mean(np.abs(empirical_coverage - levels))
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration"
    )

    ax.plot(
        levels,
        empirical_coverage,
        marker="o",
        label="LMC Multi-output GP"
    )

    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Predictive Uncertainty Calibration")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.legend()
    ax.grid(alpha=0.3)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.close(fig)

    return {
        "levels": levels,
        "empirical_coverage": empirical_coverage,
        "mace": mace
    }



def plot_calibration_curve_het(predictive, y_test, output_path, n_levels=9):
    y_test = np.asarray(y_test).squeeze()
    mean = np.asarray(predictive.mean).squeeze()
    variance = np.asarray(predictive.variance).squeeze()
    std = np.sqrt(np.clip(variance, 1e-12, None))
    levels = np.linspace(0.1, 0.9, n_levels)

    empirical_coverage = []

    for p in levels:
        a = norm.ppf((p + 1) / 2)
        lower = mean - a * std
        upper = mean + a * std
        inside = ((y_test >= lower) & (y_test <= upper))
        coverage = np.mean(inside)
        empirical_coverage.append(coverage)

    empirical_coverage = np.array(empirical_coverage)
    mace = np.mean(np.abs(empirical_coverage - levels))
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration"
    )

    ax.plot(
        levels,
        empirical_coverage,
        marker="o",
        label="GP"
    )

    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Predictive Uncertainty Calibration")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.legend()
    ax.grid(alpha=0.3)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.close(fig)

    return {
        "levels": levels,
        "empirical_coverage": empirical_coverage,
        "mace": mace
    }