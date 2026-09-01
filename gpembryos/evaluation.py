import numpy as np
import torch
import properscoring as ps
from sklearn.metrics import r2_score

# function to comptute rmse, mae, r2, nlpd, and crps
def all_metrics(y_true_standardised, predictions):

    predictions_mean = predictions.mean
    predictions_var = predictions.variance
    predictions_std = predictions.stddev

    rmse = torch.sqrt(torch.mean((predictions_mean - y_true_standardised)**2))
    mae = torch.mean(torch.abs(predictions_mean - y_true_standardised))
    r2 = r2_score(y_true_standardised.detach().cpu().numpy(), predictions_mean.detach().cpu().numpy())
    nlpd = (0.5*torch.log(2*torch.pi*predictions_var) + (y_true_standardised - predictions_mean)**2 / (2*predictions_var)).mean()
    crps = ps.crps_gaussian(y_true_standardised.detach().cpu().numpy(), mu = predictions_mean.detach().cpu().numpy(), sig = predictions_std.detach().cpu().numpy())
    crps_mean = np.mean(crps)

    return{"rmse": np.round(rmse.item(), 3),
            "mae": np.round(mae.item(), 3),
            "r2": np.round(r2, 3),
            "nlpd": np.round(nlpd.item(), 3),
            "crps": np.round(crps_mean.item(), 3)}



def multioutput_metrics(y_true_standardised, predictions, gene_names):

    predictions_mean = predictions.mean
    predictions_var = predictions.variance
    predictions_std = predictions.stddev

    y_true = y_true_standardised.detach().cpu().numpy()
    y_mean = predictions_mean.detach().cpu().numpy()
    y_var = predictions_var.detach().cpu().numpy()
    y_std = predictions_std.detach().cpu().numpy()

    metrics = {}

    for i, gene in enumerate(gene_names):

        true_i = y_true[:, i]
        mean_i = y_mean[:, i]
        var_i = y_var[:, i]
        std_i = y_std[:, i]

        rmse = np.sqrt(np.mean((mean_i - true_i) ** 2))
        mae = np.mean(np.abs(mean_i - true_i))
        r2 = r2_score(true_i, mean_i)
        nlpd = np.mean(0.5 * np.log(2 * np.pi * var_i) + (true_i - mean_i) ** 2 / (2 * var_i))
        crps = ps.crps_gaussian(true_i, mu=mean_i, sig=std_i)
        metrics[gene] = {"rmse": np.round(rmse, 3), "mae": np.round(mae, 3), "r2": np.round(r2, 3), "nlpd": np.round(nlpd, 3), "crps": np.round(np.mean(crps), 3),}

    return metrics





def get_model_parameters(model, likelihood):
    lengthscales = (model.covar_module.base_kernel.lengthscale.detach().cpu().flatten().tolist())

    return {
        "lengthscale": np.round(lengthscales, 3),
        "outputscale": np.round(model.covar_module.outputscale.item(), 3),
        "noise": np.round(likelihood.noise.item(), 3)
    }



def get_changepoint_parameters(model, likelihood):
    kernel = model.covar_module

    lengthscale_1 = (kernel.kernel_1.base_kernel.lengthscale.detach().cpu().flatten().tolist())
    outputscale_1 = (kernel.kernel_1.outputscale.detach().cpu().item())
    lengthscale_2 = (kernel.kernel_2.base_kernel.lengthscale.detach().cpu().flatten().tolist())
    outputscale_2 = (kernel.kernel_2.outputscale.detach().cpu().item())
    width = torch.exp(kernel.log_width).detach().cpu().item()
    noise = (likelihood.noise.detach().cpu().item())

    boundary_center = (kernel.boundary_center.detach().cpu().flatten().tolist())
    boundary_radius = torch.nn.functional.softplus(kernel.boundary_radius_g).item()


    return {
        "boundary_center": np.round(boundary_center, 3),
        "boundary_radius": np.round(boundary_radius, 3),
        "width": np.round(width, 3),
        "lengthscale_1": lengthscale_1,
        "lengthscale_1_x": lengthscale_1[0],
        "lengthscale_1_y": lengthscale_1[1],
        "lengthscale_1_z": lengthscale_1[2],
        "outputscale_1": outputscale_1,
        
        "lengthscale_2": lengthscale_2,
        "lengthscale_2_x": lengthscale_2[0],
        "lengthscale_2_y": lengthscale_2[1],
        "lengthscale_2_z": lengthscale_2[2],
        "outputscale_2": outputscale_2,

        "noise": noise,
    
    }





def get_multi_parameters(model, likelihood):

    lengthscales = (model.covar_module.base_kernel.lengthscale.detach().cpu().squeeze(1).tolist())
    outputscales = (model.covar_module.outputscale.detach().cpu().tolist())
    noises = (likelihood.noise.detach().cpu().tolist())

    return {
        "lengthscale": np.round(lengthscales, 3).tolist(),
        "outputscale": np.round(outputscales, 3).tolist(),
        "noise": np.round(noises, 3).tolist()
    }



def get_gneiting_parameters(model, likelihood):
    kernel = model.covar_module.base_kernel

    parameters = {
        "spatial_lengthscale": (kernel.spatial_lengthscale.detach().cpu().tolist()),
        "temporal_scale": (kernel.temporal_scale.detach().cpu().item()),
        "alpha": (kernel.alpha.detach().cpu().item()),
        "beta": (kernel.beta.detach().cpu().item()),
        "outputscale": (model.covar_module.outputscale.detach().cpu().item()),
        "noise": (likelihood.noise.detach().cpu().item())
    }

    return parameters


def get_separable_parameters(model, likelihood):

    spatial_lengthscales = (model.spatial_kernel.lengthscale.detach().cpu().numpy().flatten())

    temporal_lengthscale = (model.temporal_kernel.lengthscale.detach().cpu().item())

    outputscale = (model.covar_module.outputscale.detach().cpu().item())

    noise = (likelihood.noise.detach().cpu().item())

    return {
        "spatial_lengthscale_x": spatial_lengthscales[0],
        "spatial_lengthscale_y": spatial_lengthscales[1],
        "spatial_lengthscale_z": spatial_lengthscales[2],
        "temporal_lengthscale": temporal_lengthscale,
        "outputscale": outputscale,
        "noise": noise,
    }