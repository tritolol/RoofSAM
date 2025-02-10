from typing import Iterable

import torch
from torch import nn
import numpy as np

def majority_vote(preds: torch.Tensor) -> torch.Tensor:
    """
    Computes the majority vote for each sample (each row of predictions).
    Since torch.mode is not supported on MPS, we use NumPy.
    """
    preds_cpu = preds.cpu().numpy()  # shape: [batch_size, num_points]
    majority = []
    for sample in preds_cpu:
        counts = np.bincount(sample)
        majority.append(np.argmax(counts))
    # Return a tensor with the majority vote for each sample on the original device
    return torch.tensor(majority, device=preds.device)


def set_requires_grad(parameters: Iterable[nn.Parameter], requires_grad: bool):
    """
    Sets the `requires_grad` flag for all given PyTorch parameters.

    Args:
        parameters (Iterable[nn.Parameter]): The parameters to update.
        requires_grad (bool): Whether to enable gradient computation.

    Example:
        >>> set_requires_grad(model.parameters(), False)
    """
    for param in parameters:
        param.requires_grad = requires_grad
