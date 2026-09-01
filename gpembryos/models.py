import gpytorch as gp
import torch
import numpy as np
import torch.nn as nn

import jax
import jax.numpy as jnp
import optax as ox
import gpjax as gpx

# Exact inference GP, kernel has length-scales for each dimension
class ExactGPModel(gp.models.ExactGP):
    def __init__(self, train_inputs, train_targets, likelihood, nu = 0.5):
        super(ExactGPModel, self).__init__(train_inputs, train_targets, likelihood)
        self.mean_module = self.mean_module = gp.means.ConstantMean()
        self.covar_module = gp.kernels.ScaleKernel(gp.kernels.MaternKernel(nu=nu, ard_num_dims=3))
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gp.distributions.MultivariateNormal(mean_x, covar_x)


# Sparse Variational GP, kernel has length-scales for each dimension
class SVGP(gp.models.ApproximateGP):
    def __init__(self, inducing_points, nu):
        variational_distribution = (gp.variational.CholeskyVariationalDistribution(inducing_points.size(0)))
        variational_strategy = (gp.variational.VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True))
        super().__init__(variational_strategy)
        self.mean_module = gp.means.ConstantMean()
        self.covar_module = gp.kernels.ScaleKernel(gp.kernels.MaternKernel(nu=nu, ard_num_dims=3))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gp.distributions.MultivariateNormal(mean_x, covar_x)


# Change-point kernel as outlined in the paper, each kernel has length-scales for each dimension
# The boundary function is determined by a sphere
class ChangePointKernel(gp.kernels.Kernel):
    is_stationary = False
    def __init__(self, input_dim=3, initial_width=0.5, nu1 = 0.5, nu2 = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.kernel_1 = gp.kernels.ScaleKernel(gp.kernels.MaternKernel(nu=nu1, ard_num_dims=input_dim))
        self.kernel_2 = gp.kernels.ScaleKernel(gp.kernels.MaternKernel(nu=nu2, ard_num_dims=input_dim))
        self.boundary_center = nn.Parameter(torch.zeros(input_dim))
        self.boundary_radius_g = nn.Parameter(torch.tensor(0.0))
        self.log_width = nn.Parameter(torch.tensor(np.log(initial_width)))

    def boundary_function(self, X):
        center = self.boundary_center
        radius = torch.nn.functional.softplus(self.boundary_radius_g)

        return torch.sum((X - center) ** 2, dim=-1) - radius ** 2

    def gate(self, X):
        h = self.boundary_function(X)
        width = torch.exp(self.log_width)
        return torch.sigmoid(h / width)
    
    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        K1 = self.kernel_1(x1, x2, diag=diag, last_dim_is_batch=False)
        K2 = self.kernel_2(x1, x2, diag=diag, last_dim_is_batch=False)
        g1 = self.gate(x1)

        if diag:
            return ((1.0 - g1)**2 * K1 + g1**2 * K2)

        g2 = self.gate(x2)
        w1_x1 = (1.0 - g1).unsqueeze(-1)
        w1_x2 = (1.0 - g2).unsqueeze(-2)
        w2_x1 = g1.unsqueeze(-1)
        w2_x2 = g2.unsqueeze(-2)
        K = (w1_x1 * K1 * w1_x2 + w2_x1 * K2 * w2_x2)
        return K


class ChangePointGP(gp.models.ApproximateGP):

    def __init__(self, inducing_points, input_dim=3, nu1 = 0.5, nu2 = 0.5):
        variational_distribution = (gp.variational.CholeskyVariationalDistribution(inducing_points.size(0)))
        variational_strategy = (gp.variational.VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True))
        super().__init__(variational_strategy)
        self.mean_module = (gp.means.ConstantMean())
        self.covar_module = (ChangePointKernel(input_dim=input_dim, initial_width=0.5, nu1= nu1, nu2= nu2))

    def forward(self, x):
        mean = self.mean_module(x)
        covariance = self.covar_module(x)
        return gp.distributions.MultivariateNormal(mean, covariance)




# LMC
class MultitaskGPModel(gp.models.ApproximateGP):

    def __init__(self, inducing_points, num_tasks = 5, num_latents = 5, nu = 0.5):

        variational_distribution = (gp.variational.CholeskyVariationalDistribution(num_inducing_points=inducing_points.size(-2), batch_shape=torch.Size([num_latents])))
        variational_strategy = (gp.variational.VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True))
        strategy = (gp.variational.LMCVariationalStrategy(variational_strategy, num_tasks=num_tasks, num_latents=num_latents, latent_dim=-1))
        super().__init__(strategy)
        self.mean_module = gp.means.ConstantMean(batch_shape=torch.Size([num_latents]))
        self.covar_module = gp.kernels.ScaleKernel(gp.kernels.MaternKernel(nu=nu, ard_num_dims=3, batch_shape=torch.Size([num_latents]),), batch_shape=torch.Size([num_latents]),)

    def forward(self,x):
        mean = self.mean_module(x)
        cov = self.covar_module(x)

        return gp.distributions.MultivariateNormal(mean, cov)





# Gneiting kernel
class GneitingKernel(gp.kernels.Kernel):
    is_stationary = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.register_parameter(name="g_spatial_lengthscales", parameter=torch.nn.Parameter(torch.zeros(3)))
        self.register_constraint("g_spatial_lengthscales", gp.constraints.Positive())

        self.register_parameter(name="g_temporal_scale", parameter=torch.nn.Parameter(torch.zeros(1)))
        self.register_constraint("g_temporal_scale", gp.constraints.Positive())

        self.register_parameter(name="g_alpha", parameter=torch.nn.Parameter(torch.zeros(1)))
        self.register_constraint("g_alpha", gp.constraints.Interval(1e-6, 1.0))

        self.register_parameter(name="g_beta", parameter=torch.nn.Parameter(torch.zeros(1)))
        self.register_constraint("g_beta", gp.constraints.Interval(1e-6, 1.0))

    @property
    def spatial_lengthscale(self):
        return self.g_spatial_lengthscales_constraint.transform(self.g_spatial_lengthscales)

    @property
    def temporal_scale(self):
        return self.g_temporal_scale_constraint.transform(self.g_temporal_scale)

    @property
    def alpha(self):
        return self.g_alpha_constraint.transform(self.g_alpha)

    @property
    def beta(self):
        return self.g_beta_constraint.transform(self.g_beta)

    def forward(self, x1, x2, diag=False, **params):

        s1 = x1[..., :-1]
        s2 = x2[..., :-1]

        t1 = x1[..., -1:]
        t2 = x2[..., -1:]

        s1_ = s1/self.spatial_lengthscale
        s2_ = s2/self.spatial_lengthscale
        s_dist_sq = self.covar_dist(s1_, s2_, square_dist=True, diag=diag, **params)
        t_dist_sq = self.covar_dist(t1, t2, square_dist=True, diag=diag, **params)
        psi = (1.0 + self.temporal_scale * torch.pow(t_dist_sq + 1e-12, self.alpha)).pow(self.beta)
        d = s1.shape[-1]
        prefactor = psi.pow(-d / 2.0)
        s_dist = torch.sqrt(s_dist_sq + 1e-12)
        exponent = -s_dist / (torch.sqrt(psi))

        return prefactor * torch.exp(exponent)



class GneitingGP(gp.models.ApproximateGP):
    def __init__(self, inducing_points):

        variational_distribution = (gp.variational.CholeskyVariationalDistribution(inducing_points.size(0)))
        variational_strategy = (gp.variational.VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True))
        super().__init__(variational_strategy)
        self.mean_module = gp.means.ConstantMean()
        self.covar_module = gp.kernels.ScaleKernel(GneitingKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gp.distributions.MultivariateNormal(mean_x, covar_x,)


class SeparableSpatiotemporalGP(gp.models.ApproximateGP):

    def __init__(self, inducing_points):

        variational_distribution = (gp.variational.CholeskyVariationalDistribution(inducing_points.size(0)))
        variational_strategy = (gp.variational.VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True))
        super().__init__(variational_strategy)
        self.mean_module = gp.means.ConstantMean()
        self.spatial_kernel = gp.kernels.MaternKernel(nu=0.5, ard_num_dims=3, active_dims=torch.tensor([0, 1, 2]))
        self.temporal_kernel = gp.kernels.MaternKernel(nu=0.5, ard_num_dims=1, active_dims=torch.tensor([3]))
        self.covar_module = gp.kernels.ScaleKernel(self.spatial_kernel * self.temporal_kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gp.distributions.MultivariateNormal(mean_x, covar_x)