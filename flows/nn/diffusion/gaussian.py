from jaxtyping import Float
import torch
from torch import nn, Tensor

from diffusion_sandbox.nn.noise_model import NoiseModel
from diffusion_sandbox.nn.schedule import VarianceSchedule


class GaussianDiffusion(nn.Module):
    def __init__(self, model: NoiseModel, schedule: VarianceSchedule):
        super().__init__()

        self.model = model
        self.schedule = schedule

    def noise(self, x_0: Float[Tensor, "b ... d"], t: int):
        eps = torch.randn_like(x_0)
        alpha_bar = self.schedule.alpha_bar[t].expand_as(x_0)

        return alpha_bar.sqrt() * x_0 + (1 - alpha_bar).sqrt() * eps

    def denoise(self, model, x_t, t):
        beta, alpha, alpha_bar = self.schedule.alpha[0]
        eps_hat = model(x_t, t)
        mu_0_tilde = (1 / alpha.sqrt()) * (x_t - beta / (1 - alpha_bar).sqrt() * eps_hat)

        return mu_0_tilde

    def sample(self, model: NoiseModel, n):
        x_t = torch.randn(n, model.input_dim)
        for t in range(len(self.schedule) - 1, 0, -1):
            mu_0_tilde = self.denoise(model, x_t, t)
            eps = self.schedule.beta[t].sqrt() * torch.randn_like(x_t)

            x_t = mu_0_tilde + eps
        return self.denoise(model, x_t, 0)


