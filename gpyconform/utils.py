#!/usr/bin/env python3

import math
import torch

def _tensor_split(
    X: torch.Tensor,
    y: torch.Tensor,
    size: int | float,
    *,
    size_name: str,
    shuffle: bool = True,
    seed: int = None,
):
    if X.size(0) != y.size(0):
        raise ValueError(
            f"X and y must have the same number of samples; got {X.size(0)} and {y.size(0)}."
        )
    N = X.size(0)
    if N < 2:
        raise ValueError("Need at least 2 samples to split.")

    # Resolve split count (k)
    if isinstance(size, float):
        if not (0.0 < size < 1.0):
            raise ValueError(f"When float, {size_name} must be in (0, 1).")
        k = int(math.ceil(N * size))
        k = max(1, min(k, N - 1))
    elif isinstance(size, int):
        if not (1 <= size <= N - 1):
            raise ValueError(f"When int, {size_name} must be in [1, {N-1}] for n_samples={N}.")
        k = size
    else:
        raise TypeError(f"{size_name} must be int or float.")

    # Build indices
    if shuffle:
        if seed is None:
            perm = torch.randperm(N, device="cpu")
        else:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(seed)
            perm = torch.randperm(N, generator=gen, device="cpu")
    else:
        perm = torch.arange(N, device="cpu")

    train_idx = perm[:-k]
    test_idx = perm[-k:]

    # Index-select along the first dimension; stays on same device/dtype
    X_train = X.index_select(0, train_idx.to(X.device))
    X_test  = X.index_select(0, test_idx.to(X.device))
    y_train = y.index_select(0, train_idx.to(y.device))
    y_test  = y.index_select(0, test_idx.to(y.device))

    return X_train, X_test, y_train, y_test


def tensor_train_test_split(
    X: torch.Tensor,
    y: torch.Tensor,
    test_size: int | float,
    *,
    shuffle: bool = True,
    seed: int = None,
):
    """
    Split paired tensors (X, y) into train and test sets (non-overlapping along dim 0).

    Parameters
    ----------
    X : torch.Tensor of shape (n_samples, n_features)
        Features.
    y : torch.Tensor of shape (n_samples,)
        Targets. Must have same first dimension as X.
    test_size : int or float
        If int, the number of samples for the test set (integer in [1, n_samples-1]).
        If float, the fraction in (0, 1) for the test set.
    shuffle : bool, default=True
        Whether to shuffle before splitting.
    seed : int or None, default=None
        Random seed used when shuffling for reproducibility.

    Returns
    -------
    X_train : torch.Tensor
        Training set features.
    X_test : torch.Tensor
        Test set features.
    y_train : torch.Tensor
        Training set targets.
    y_test : torch.Tensor
        Test set targets.

    Notes
    -----
    - Works entirely on the original devices/dtypes of X and y.
    - A single CPU permutation is generated (optionally seeded) and the sliced
      indices are moved to each tensor's device before indexing.
    - For float test_size, uses ceil(n_samples * test_size) clamped to [1, n_samples-1].
    
    Examples
    --------
    Assuming ``all_data_x`` and ``all_data_y`` are torch tensors with all the dataset 
    features and targets respectively, they can be divided into training and test tensors 
    with 30% of the samples assigned to the test set and the remaining 70% to the training
    set by:

    >>> train_x, test_x, train_y, test_y = tensor_train_test_split(all_data_x, all_data_y, test_size=0.3, seed=123)

    """
    return _tensor_split(X, y, test_size, size_name="test_size", shuffle=shuffle, seed=seed)


def tensor_train_cal_split(
    X: torch.Tensor,
    y: torch.Tensor,
    cal_size: int | float,
    *,
    shuffle: bool = True,
    seed: int = None,
):
    """
    Split training set tensors (X, y) into proper-training and calibration sets (non-overlapping along dim 0).

    Parameters
    ----------
    X : torch.Tensor of shape (n_samples, n_features)
        Features.
    y : torch.Tensor of shape (n_samples,)
        Targets. Must have same first dimension as X.
    cal_size : int or float
        If int, number of samples for the calibration set (integer in [1, n_samples-1]).
        If float, fraction in (0, 1) for the calibration set.
    shuffle : bool, default=True
        Whether to shuffle before splitting.
    seed : int or None, default=None
        Random seed used when shuffling for reproducibility.

    Returns
    -------
    X_prop_train : torch.Tensor
        Proper training set features.
    X_cal : torch.Tensor
        Calibration set features.
    y_prop_train : torch.Tensor
        Proper training set targets.
    y_cal : torch.Tensor
        Calibration set targets.

    Notes
    -----
    - Works entirely on the original devices/dtypes of X and y.
    - A single CPU permutation is generated (optionally seeded) and the sliced
      indices are moved to each tensor's device before indexing.
    - For float cal_size, uses ceil(n_samples * cal_size) clamped to [1, n_samples-1].
    
    Examples
    --------
    Assuming ``train_x`` and ``train_y`` are torch tensors with the training set 
    features and targets respectively, they can be divided into proper-training and 
    calibration tensors with 25% of the training samples assigned to the calibration 
    set and the remaining 75% to the proper-training set by:

    >>> prop_train_x, cal_x, prop_train_y, cal_y = tensor_train_cal_split(train_x, train_y, cal_size=0.25, seed=123)

    """
    X_prop_train, X_cal, y_prop_train, y_cal = _tensor_split(X, y, cal_size, size_name="cal_size", shuffle=shuffle, seed=seed)

    return X_prop_train, X_cal, y_prop_train, y_cal
