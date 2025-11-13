import torch
import torch.nn as nn

# NOT ACTUALLY CHECKED THIS IMPLEMENTATION YET, UPLOADED FOR A COLLEAGUE
class IsotropicReLU(nn.Module):
    def __init__(self):
        super(IsotropicReLU, self).__init__()
        self.R = 1

    def forward(self, x: torch.Tensor, dims:tuple[int, ...]=(-1,))-> torch.Tensor:
        # Calculate the vector magnitude
        magnitude = torch.linalg.norm(x, dim=dims, keepdim=True)
        # A small ball approximates the identity map for more stable computation
        identity_mask = magnitude <= self.R

        # Calculate the required unit-vector
        unit_vector = x / magnitude.clamp(min=self.epsilon)
        # Return final computation, identity map for when the magnitude is small, otherwise Isotropic Tanh
        return torch.where(identity_mask, 0*x, torch.maximum(magnitude-self.R, 0) * unit_vector)
