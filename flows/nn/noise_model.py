from jaxtyping import Float
from torch import Tensor, nn


class NoiseModel(nn.Module):
    input_dim: int

    def forward(self, x: Float[Tensor, "b d"], t: int) -> Float[Tensor, "b d"]:
        pass