"""Torch datasets and transforms used by model training."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import Dataset


class Anscombe(nn.Module):
    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return 2.0 * torch.sqrt(values + 3.0 / 8.0)

    def inv(self, values: torch.Tensor) -> torch.Tensor:
        return (values / 2.0) ** 2 - 3.0 / 8.0


class NeuralDataset(Dataset):
    """Trials with positional, dense-task and sparse-task inputs."""

    def __init__(self, x_position_vars, x_dense_vars, x_sparse_vars, targets, mean):
        self.x_position_vars = torch.as_tensor(x_position_vars, dtype=torch.float32).transpose(1, 2)
        self.x_dense_vars = torch.as_tensor(x_dense_vars, dtype=torch.float32)
        self.x_sparse_vars = torch.as_tensor(x_sparse_vars, dtype=torch.float32)
        target_tensor = torch.as_tensor(targets, dtype=torch.float32).transpose(1, 2)
        self.Y = Anscombe()(target_tensor) - torch.as_tensor(mean).detach().cpu()

    def __len__(self) -> int:
        return self.x_position_vars.size(0)

    def __getitem__(self, index: int):
        return (self.x_position_vars[index], self.x_dense_vars[index], self.x_sparse_vars[index]), self.Y[index]
