import torch
import torch.nn as nn

class IsotropicTanh(nn.Module):
    def __init__(self):
        super(IsotropicTanh, self).__init__()

        # Since tanh approximates the identity around the origin, then a small ball is just treated as the identity map this ensures computational stability. This is because there is a coordinate artefact about the centre, which shouldn't but does meaningfully affect computation due to precision. Therefore, avoid this with this threshold.
        self.epsilon = 1e-2

    def forward(self, x: torch.Tensor, dims:tuple[int, ...]=(-1,))-> torch.Tensor:
        # Calculate the vector magnitude
        magnitude = torch.linalg.norm(x, dim=dims, keepdim=True)
        # A small ball approximates the identity map for more stable computation
        identity_mask = magnitude <= self.epsilon
        # Calculate the required unit-vector
        unit_vector = x / magnitude.clamp(min=self.epsilon)
        # Return final computation, identity map for when the magnitude is small, otherwise Isotropic Tanh
        return torch.where(identity_mask, x, torch.tanh(magnitude) * unit_vector)
