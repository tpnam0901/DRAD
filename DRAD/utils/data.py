import numpy as np
import torch
from typing import Union, List


def min_max_normalize_dataset(data: Union[np.ndarray, torch.Tensor], type_data: str) -> Union[np.ndarray, torch.Tensor]:
    """Apply max normalization to the input data with predefined columns and max values.
    Max Volt: 46.172
    Max Current: 180.78
    Max Max Single Volt: 4.2309
    Max Min Single Volt: 4.2151
    Max Max Temp: 39.0
    Max Min Temp: 36.0

    Min Volt: 36.20217999999999
    Min Current: -180.78
    Min Max Single Volt: 3.4252
    Min Min Single Volt: 3.3384
    Min Max Temp: -4.0
    Min Min Temp: -4.0
    Args:
        data (Union[np.ndarray, torch.Tensor]): Input data to be normalized.
        type_data (str): Type of data columns.
        dim (int): Dimension along which is normalized.
    Returns:
        Union[np.ndarray, torch.Tensor]: Min-Max normalized data.
    """
    assert len(data.shape) == 1, "Input data must be 1-dimensional."

    max_values = {
        "volt": 46.172,
        "current": 180.78,
        "soc": 100.0,
        "max_single_volt": 4.2309,
        "min_single_volt": 4.2151,
        "max_temp": 39.0,
        "min_temp": 36.0,
    }
    min_values = {
        "volt": 36.20217999999999,
        "current": -180.78,
        "soc": 0.0,
        "max_single_volt": 3.4252,
        "min_single_volt": 3.3384,
        "max_temp": -4.0,
        "min_temp": -4.0,
    }
    if type_data not in max_values:
        raise ValueError(f"Unsupported type '{type_data}' for min-max normalization.")
    max_val = max_values[type_data]
    min_val = min_values[type_data]
    if isinstance(data, np.ndarray):
        normalized_data = (data - min_val) / (max_val - min_val + 1e-8)
    elif isinstance(data, torch.Tensor):
        normalized_data = (data - min_val) / (max_val - min_val + 1e-8)
    else:
        raise TypeError("Input data must be either a numpy array or a torch tensor.")
    return normalized_data


def min_max_mileage_normalize(data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """Apply min-max normalization to mileage data.
    Min Mileage: 0
    Max Mileage: 44362.376520000005

    Args:
        data (Union[np.ndarray, torch.Tensor]): Input mileage data to be normalized.
    Returns:
        Union[np.ndarray, torch.Tensor]: Min-max normalized mileage data.
    """
    max_mileage = 44362.376520000005
    min_mileage = 0.0
    if isinstance(data, np.ndarray):
        normalized_data = (data - min_mileage) / (max_mileage - min_mileage)
    elif isinstance(data, torch.Tensor):
        normalized_data = (data - min_mileage) / (max_mileage - min_mileage)
    else:
        raise TypeError("Input data must be either a numpy array or a torch tensor.")
    return normalized_data


def min_max_normalize(data: Union[np.ndarray, torch.Tensor], dim: int = 0) -> Union[np.ndarray, torch.Tensor]:
    """Apply min-max normalization to the input data.

    Args:
        data (Union[np.ndarray, torch.Tensor]): Input data to be normalized.
        dim (int): Dimension along which to compute the min and max.
    Returns:
        Union[np.ndarray, torch.Tensor]: Min-max normalized data.
    """
    if isinstance(data, np.ndarray):
        min_val = np.min(data, axis=dim, keepdims=True)
        max_val = np.max(data, axis=dim, keepdims=True)
        normalized_data = (data - min_val) / (max_val - min_val + 1e-8)
    elif isinstance(data, torch.Tensor):
        min_val = torch.min(data, dim=dim, keepdim=True)[0]
        max_val = torch.max(data, dim=dim, keepdim=True)[0]
        normalized_data = (data - min_val) / (max_val - min_val + 1e-8)
    else:
        raise TypeError("Input data must be either a numpy array or a torch tensor.")
    return normalized_data


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
        std = torch.std(data, dim=dim, keepdim=True)
        normalized_data = (data - mean) / (std + 1e-8)
    else:
        raise TypeError("Input data must be either a numpy array or a torch tensor.")
    return normalized_data


def padding_to_max_length(data: torch.Tensor, max_length: int, dim: int = 0, padding_value: float = 0) -> torch.Tensor:
    """Pad the input tensor to the specified maximum length along a given dimension.

    Args:
        data (torch.Tensor): Input tensor to be padded.
        max_length (int): Desired maximum length after padding.
        dim (int): Dimension along which to pad the tensor. Default is 0.
        padding_value (float): Value to use for padding. Default is 0.
    Returns:
        torch.Tensor: Padded tensor with the specified maximum length.
    """
    current_length = data.size(dim)
    if current_length >= max_length:
        return data
    pad_size = list(data.shape)
    pad_size[dim] = max_length - current_length
    padding = torch.full(pad_size, padding_value, dtype=data.dtype, device=data.device)
    padded_data = torch.cat([data, padding], dim=dim)
    return padded_data


def random_crop_tensor(data: Union[np.ndarray, torch.Tensor], size: int, dim: int = 0) -> torch.Tensor:
    """Randomly crop the input tensor into smaller tensors of specified size along a given dimension.

    Args:
        data (torch.Tensor): Input tensor to be cropped.
        size (int): Size of each crop.
        dim (int): Dimension along which to crop the tensor. Default is 0.
    Returns:
        torch.Tensor: Cropped tensor of specified size.
    """
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    current_length = data.size(dim)
    if current_length <= size:
        return data
    start = np.random.randint(0, current_length - size + 1)
    end = start + size
    if end > current_length:
        end = current_length
        start = end - size
    cropped_data = data.narrow(dim, start, size)
    return cropped_data


def chunk_tensor_with_overlap(data: Union[np.ndarray, torch.Tensor], chunk_size: int, overlap: float, dim: int = 0) -> list[torch.Tensor]:
    """Chunk the input tensor into smaller tensors of specified size with overlap along a given dimension.

    Args:
        data (Union[np.ndarray, torch.Tensor]): Input tensor to be chunked.
        chunk_size (int): Size of each chunk.
        overlap (float): Fraction of overlap between consecutive chunks (0.0 to 1.0).
        dim (int): Dimension along which to chunk the tensor. Default is 0.
    Returns:
        list[torch.Tensor]: List of chunked tensors with overlap.
    """
    step_size = int(chunk_size * (1 - overlap))
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    chunks = []
    start = 0
    while start < data.shape[dim]:
        end = min(start + chunk_size, data.shape[dim])
        if isinstance(data, torch.Tensor):
            chunk = data.narrow(dim, start, end - start)
        else:
            slice_obj = [slice(None)] * data.ndim
            slice_obj[dim] = slice(start, end)
            chunk = data[tuple(slice_obj)]
        chunks.append(chunk)
        if end == data.size(dim):
            break
        start += step_size
    return chunks
