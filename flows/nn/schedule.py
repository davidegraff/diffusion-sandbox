from jaxtyping import ArrayLike, Float
import torch
from torch import Tensor, nn


class VarianceSchedule(nn.Module):
    beta: Float[Tensor, "t"]
    alpha: Float[Tensor, "t"]
    alpha_bar: Float[Tensor, "t"]

    def __init__(self, betas: Float[ArrayLike, "t"]):
        super().__init__()

        beta = torch.as_tensor(betas)
        alpha = 1 - beta
        alpha_bar = alpha.cumprod()

        if not (beta.diff() >= 0).all():
            raise ValueError("arg 'betas' must be strictly increasing!")

        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_bar", alpha_bar)
    
    def __len__(self):
        return len(self.beta)

    def __getitem__(self, index: int) -> tuple[float, float, float]:
        return self.beta[index], self.alpha[index], self.alpha[index]

    def extra_repr(self):
        return f"(beta): {self.beta}"
