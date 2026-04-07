from typing import Tuple, Union

import numpy as np
import torch


def z_score_normalize(data: Union[np.ndarray, torch.Tensor], dim: int = 0) -> Union[np.ndarray, torch.Tensor]:
    """Apply z-score normalization to the input data.

    Args:
        data (Union[np.ndarray, torch.Tensor]): Input data to be normalized.
        dim (int): Dimension along which to compute the mean and std.
    Returns:
        Union[np.ndarray, torch.Tensor]: Z-score normalized data.
    """
    if isinstance(data, np.ndarray):
        mean = np.mean(data, axis=dim, keepdims=True)
        std = np.std(data, axis=dim, keepdims=True)
        normalized_data = (data - mean) / (std + 1e-8)
    elif isinstance(data, torch.Tensor):
        mean = torch.mean(data, dim=dim, keepdim=True)

        normalized_data = (data - mean) / (std + 1e-8)
    else:
        raise TypeError("Input data must be either a numpy array or a torch tensor.")
    return normalized_data


def get_data_mean_std(
    data: Union[np.ndarray, torch.Tensor], dim: int = 0
) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
    if isinstance(data, np.ndarray):
        mean = np.mean(data, axis=dim, keepdims=True)
        std = np.std(data, axis=dim, keepdims=True)
    elif isinstance(data, torch.Tensor):
        mean = torch.mean(data, dim=dim, keepdim=True)
        std = torch.std(data, dim=dim, keepdim=True)
    else:
        raise TypeError("Input data must be either a numpy array or a torch tensor.")
    return mean, std
