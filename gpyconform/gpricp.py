#!/usr/bin/env python3

import torch
import numpy as np
import warnings
import hashlib

from torch.utils.data import DataLoader, TensorDataset
from gpytorch.models import GP
from gpyconform.prediction_intervals import PredictionIntervals
from gpyconform.constants import _CONF_SCALE, _GAMMA_INF_THRESHOLD

@torch.no_grad()
def _predict_gp(model, test_inputs, batch_size=None, device=None, dtype=None):
    """
    Compute predictive means & variances for a trained GPyTorch model.

    Parameters
    ----------
    model : gpytorch.models.GP
        Trained GP model. The function will set the model (and ``model.likelihood`` if present)
        to ``.eval()`` mode for prediction.
    test_inputs : torch.Tensor or torch.utils.data.DataLoader
        Test inputs as a single Tensor (shape (n_test, n_features) or (n_test, ...)), or a DataLoader
        yielding Tensor batches. If a DataLoader yields tuples, only the first
        element is used as inputs; extras are ignored with a warning.
    batch_size : int, optional
        If ``test_inputs`` is a Tensor, optionally split into batches of this size.
    device : torch.device or str, optional
        If provided, move inputs (or each batch) to this device before forward.
    dtype : torch.dtype, optional
        If provided, cast inputs (or each batch) to this dtype before forward.

    Returns
    -------
    means : torch.Tensor of shape (n_test,)
        Predictive means.
    variances : torch.Tensor of shape (n_test,)
        Predictive variances.

    Notes
    -----
    - No gradients are computed. 
    - The function puts the model (and likelihood, if present) in ``.eval()`` mode 
        and does not restore the previous training/eval state.
    - The function honors the model's own dtype/device: 
        it only casts inputs to ``device``/``dtype`` if specified.
    """

    model.eval()
    if hasattr(model, "likelihood"):
        model.likelihood.eval()

    # If a Tensor and no batching -> single-shot
    if isinstance(test_inputs, torch.Tensor) and batch_size is None:
        if device is not None or dtype is not None:
            test_inputs = test_inputs.to(device=device or test_inputs.device,
                     dtype=dtype or test_inputs.dtype,
                     non_blocking=True)
        f_post = model(test_inputs)
        obs_post = model.likelihood(f_post) if hasattr(model, "likelihood") else f_post
        return obs_post.mean, obs_post.variance

    # Otherwise, build/accept a DataLoader
    if isinstance(test_inputs, torch.Tensor):
        ds = TensorDataset(test_inputs)
        loader = DataLoader(ds, batch_size=batch_size or len(test_inputs))
    else:
        # assume user passed in a DataLoader of (x_batch,) tuples
        loader = test_inputs

    all_means, all_vars = [], []
    for batch in loader:
        if torch.is_tensor(batch):
            x_batch = batch
        elif isinstance(batch, (list, tuple)) and len(batch) >= 1 and torch.is_tensor(batch[0]):
            x_batch = batch[0]
            if len(batch) > 1:  # extras exist
                warnings.warn("DataLoader contains extra items (e.g., targets); ignoring extras.", RuntimeWarning)
        else:
            raise TypeError("DataLoader must yield a Tensor or a tuple/list whose first element is a Tensor.")

        if device is not None or dtype is not None:
            x_batch = x_batch.to(device=device or x_batch.device,
                                 dtype=dtype or x_batch.dtype,
                                 non_blocking=True)
        f_post = model(x_batch)
        obs_post = model.likelihood(f_post) if hasattr(model, "likelihood") else f_post
        all_means.append(obs_post.mean)
        all_vars.append(obs_post.variance)

    return torch.cat(all_means, dim=0), torch.cat(all_vars, dim=0)

def _normalize_device(dev: torch.device | str) -> torch.device:
    if isinstance(dev, str):
        dev = torch.device(dev)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is not available.")
    if dev.type == "cuda" and dev.index is None:
        # pick cuda:0 by default
        dev = torch.device("cuda:0")
    return dev


class InductiveConformalRegressor:
    """
    Inductive conformal regressor calibrated on a given calibration set.

    This class stores calibration residuals (normalized or not) computed from
    a calibration set and produces prediction intervals (PIs) for new points.
    It can be used not only with GPyTorch models, but with any regression framework
    that provides predictive means and variances.

    Parameters
    ----------
    cal_targets : torch.Tensor of shape (n_cal,)
        Calibration targets.
    cal_preds : torch.Tensor of shape (n_cal,)
        Predictive means at the calibration inputs.
    cal_vars : torch.Tensor of shape (n_cal,), optional
        Predictive variances at the calibration inputs. 
        If ``None``, the non-normalized residuals are used as nonconformity 
        measure (equivalent to setting ``gamma = ∞``).
    gamma : float, optional, default=2.0
        Power parameter for the normalized ICP nonconformity. If ``gamma >= 1e8``,
        normalization is short-circuited (equivalent to unnormalized nonconformity).
    cpmode : {'symmetric', 'asymmetric'}, optional, default='symmetric'
        Conformal mode. ``'symmetric'`` uses absolute residuals; ``'asymmetric'``
        uses signed residuals. (``None`` is not accepted here.)
    device : torch.device or str, optional
        Device for internal tensors. Defaults to the device inferred from
        the inputs.

    Notes
    -----
    - All ICP computations inside this class are performed in **float64**.
    - A small epsilon clamp (``1e-12``) is used when taking powers of variances.
    - The calibrated nonconformity scores (and therefore the resulting PIs) are
      tied to the chosen ``cpmode`` and ``gamma``. For a different ``cpmode`` or
      ``gamma``, construct a new :class:`InductiveConformalRegressor`.

    Examples
    --------
    Assuming ``cal_x`` and ``cal_y`` are torch tensors with the calibration set 
    inputs and targets, and ``cal_means`` and ``cal_vars`` are torch tensors with 
    the predictive means and variances at ``cal_x``. An ``InductiveConformalRegressor`` 
    with the absolute residual nonconformity measure (``cpmode='symmetric'``) and 
    ``gamma`` set to 2 can be constructed by:
    
    >>> calibratedICR = InductiveConformalRegressor(
    ...     cal_y, cal_means, cal_vars, gamma=2.0, cpmode='symmetric'
    ... )

    """

    def __init__(self, cal_targets, cal_preds, cal_vars=None, gamma=2.0, cpmode='symmetric',
                 device: torch.device | str | None = None):

        if cpmode not in ['symmetric', 'asymmetric']:
            raise ValueError("Invalid cpmode. Valid values are 'symmetric' or 'asymmetric'.")

        self._dtype = torch.float64

        if device is not None:
            self._device = _normalize_device(device)
        else:
            self._device = cal_preds.device            
            
        cal_targets = cal_targets.to(device=self._device, dtype=self._dtype, non_blocking=True)
        cal_preds = cal_preds.to(device=self._device, dtype=self._dtype, non_blocking=True)
        
        if gamma >= _GAMMA_INF_THRESHOLD:
            self.inv_gamma = None
        else:
            self.inv_gamma = torch.as_tensor(1.0/float(gamma), device=self._device, dtype=self._dtype)
        
        self.cpmode = cpmode
        if cal_vars is None:
            self.normalized = False
            if cpmode == 'symmetric':
                self.alphas, _ = torch.sort(torch.abs(cal_targets - cal_preds), descending=True)
            else:
                self.alphas, _ = torch.sort((cal_targets - cal_preds), descending=True)                
        else:
            self.normalized = True
            eps = 1e-12
            if self.inv_gamma is None:
                den = torch.ones_like(cal_vars, device=self._device, dtype=self._dtype)
            else:
                cal_vars = cal_vars.to(device=self._device, dtype=self._dtype, non_blocking=True)
                den = cal_vars.clamp_min(eps).pow(self.inv_gamma)
            
            if cpmode == 'symmetric':
                self.alphas, _ = torch.sort(torch.abs(cal_targets - cal_preds) / den, descending=True)
            else:
                self.alphas, _ = torch.sort((cal_targets - cal_preds) / den, descending=True)
                
    def __call__(self, test_preds, test_vars=None, confs=None):
        """
        Produce prediction intervals at requested confidence levels.

        Parameters
        ----------
        test_preds : torch.Tensor of shape (n_test,)
            Predictive means at the test inputs.
        test_vars : torch.Tensor of shape (n_test,), optional
            Predictive variances at the test inputs.
            Required if ``cal_vars`` were provided at construction time; 
            ignored otherwise.
        confs : array-like of float in (0, 1), optional, default=[0.95]
            Confidence levels for which to return PIs.

        Returns
        -------
        PredictionIntervals : gpyconform.PredictionIntervals
            Prediction intervals for each confidence level in ``confs``. The 
            returned object keeps the intervals on the same device as this
            ``InductiveConformalRegressor`` instance.

        Examples
        --------
        Assuming ``calibratedICR`` is an ``InductiveConformalRegressor``, and 
        ``test_means`` and ``test_vars`` are torch tensors with the predictive means 
        and variances at the test inputs. The Conformal Prediction Intervals at the
        90%, 95%, and 99% confidence levels can be obtained as an instance of 
        ``PredictionIntervals`` by:
    
        >>> PIs = calibratedICR(test_means, test_vars, confs=[0.9, 0.95, 0.99])

        """

        if confs is None:
            confs = torch.tensor([0.95], device=self._device, dtype=self._dtype)
        else:
            if not isinstance(confs, (torch.Tensor, np.ndarray, list, tuple)):
                raise TypeError(
                    f"'confs' must be a torch.Tensor, numpy.ndarray, list, or tuple, "
                    f"but got {type(confs).__name__}"
                    )

            confs = torch.as_tensor(confs, device=self._device, dtype=self._dtype)

            if confs.numel() == 0:
                raise ValueError("'confs' must contain at least one confidence level.")

            if not torch.all((confs > 0) & (confs < 1)):
                raise ValueError(
                    f"All confidence levels must be strictly between 0 and 1, but got: {confs}"
                    )

        test_preds = test_preds.to(device=self._device, dtype=self._dtype, non_blocking=True)

        if self.normalized:
            if test_vars is None:
                raise ValueError(
                    "The normalized nonconformity measure requires 'test_vars'"
                    )
            else:
                if self.inv_gamma is None:
                    sigmas = torch.ones_like(test_vars, device=self._device, dtype=self._dtype)
                else:
                    test_vars = test_vars.to(device=self._device, dtype=self._dtype, non_blocking=True)
                    eps = 1e-12
                    sigmas = test_vars.clamp_min(eps).pow(self.inv_gamma)
        else:
            sigmas = torch.ones_like(test_preds, device=self._device, dtype=self._dtype)
            if test_vars is not None:
                warnings.warn(
                    "'test_vars' provided but not used, since 'cal_vars' was not provided in '__init__'", 
                    RuntimeWarning 
                )

        confs_int = torch.round(confs * _CONF_SCALE).to(torch.int64)

        cal_size = self.alphas.shape[0]
        if self.cpmode == 'symmetric':
            alpha_idxs = ((_CONF_SCALE - confs_int) * (cal_size + 1)) // _CONF_SCALE - 1

            if (alpha_idxs < 0).any():
                warnings.warn(
                    "Calibration set too small for at least one confidence level; the intervals will be of maximum size for those levels.",
                    RuntimeWarning
                )

            neg_mask = alpha_idxs < 0
            alpha_idxs_safe = alpha_idxs.clamp_min(0)

            confs_alphas = self.alphas.index_select(0, alpha_idxs_safe)
            # Assign +Inf when idx < 0
            confs_alphas = torch.where(
                neg_mask,
                torch.full_like(confs_alphas, float('inf')),
                confs_alphas
                )                                                                

            unnorm_alphas = confs_alphas.unsqueeze(1) * sigmas.unsqueeze(0)

            test_preds_2d = test_preds.unsqueeze(0)
            lower = test_preds_2d - unnorm_alphas
            upper = test_preds_2d + unnorm_alphas

            PIs = torch.stack((lower, upper), dim=-1)
        else:
            alpha_idxs = (((_CONF_SCALE - confs_int) * (cal_size + 1)) // (2 * _CONF_SCALE) - 1).to(torch.long)

            if (alpha_idxs < 0).any():
                warnings.warn(
                    "Calibration set too small for at least one confidence level; the intervals will be of maximum size for those levels.",
                    RuntimeWarning
                )

            neg_mask = alpha_idxs < 0
            alpha_idxs_safe = alpha_idxs.clamp_min(0)

            confs_alphas_upper = self.alphas.index_select(0, alpha_idxs_safe)
            confs_alphas_lower_idx = (cal_size - 1) - alpha_idxs_safe
            confs_alphas_lower = self.alphas.index_select(0, confs_alphas_lower_idx)

            confs_alphas_upper = torch.where(
                neg_mask, 
                torch.full_like(confs_alphas_upper, float('inf')), 
                confs_alphas_upper
                )
            confs_alphas_lower = torch.where(
                neg_mask, 
                torch.full_like(confs_alphas_lower, float('-inf')), 
                confs_alphas_lower
                )

            # widths
            unnorm_alphas_upper = confs_alphas_upper.unsqueeze(1) * sigmas.unsqueeze(0)
            unnorm_alphas_lower = confs_alphas_lower.unsqueeze(1) * sigmas.unsqueeze(0)

            test_preds_2d = test_preds.unsqueeze(0)
            lower = test_preds_2d + unnorm_alphas_lower
            upper = test_preds_2d + unnorm_alphas_upper

            PIs = torch.stack((lower, upper), dim=-1)
            
        return PredictionIntervals(confs, PIs)
   
   
class GPRICPWrapper:
    """
    Inductive Conformal Prediction wrapper for a trained GPyTorch GP model.

    This wrapper preserves the GP model's dtype for predictions and (optionally)
    moves the model to a chosen device. All ICP computations are performed in
    float64 for numerical stability. It supports both symmetric and asymmetric 
    ICP and can reuse cached predictions.
    
    Parameters
    ----------
    model : gpytorch.models.GP
        Trained GPyTorch model. The model's dtype is **not** modified by this wrapper.
        Ensure the model (and likelihood) are in ``.eval()`` mode before 
        calibrating and predicting.
    cal_inputs : torch.Tensor of shape (n_cal, n_features) | torch.utils.data.DataLoader
        Calibration features (tensor or DataLoader yielding tensors).
        Used to compute calibration predictions.
    cal_targets : torch.Tensor of shape (n_cal,)
        Calibration targets.
    cpmode : {'symmetric', 'asymmetric', None}, optional, default='symmetric'
        Mode of the Conformal Prediction: 
        - ``'symmetric'``: Employs the absolute residual nonconformity measure approach as described in [1].
        - ``'asymmetric'``: Employs the asymmetric version of the nonconformity measure defined in [1], following the approach described in Chapter 2.3 of [2].
        - ``None``: Reverts to the provided model behavior; :meth:`predict` returns ``(means, vars)``.
    device : torch.device or str, optional
        Device to place the model on. If omitted, the current model device is used.
    batch_size : int, optional
        Batch size used when evaluating the calibration predictions from a tensor.
        If omitted, a single batch with all instances is used.

    Notes
    -----
    - If a DataLoader is provided for ``cal_inputs``, ensure it yields batches in a
      fixed order (``shuffle=False``) so predictions align with ``cal_targets``.
    - Inputs passed to the GP are cast to the **model's** dtype/device.
    - All ICP calculations run in **float64** for numerical stability.
    - Calibration depends on the current ``cpmode`` and the chosen ``gamma``.
      If either changes, call :meth:`calibrate` again.
    - When ``cpmode=None``, this wrapper returns ``(means, vars)`` tensors
      (not a ``MultivariateNormal`` like :class:`gpyconform.ExactGPCP`).
    - ``__call__`` is an alias of :meth:`predict`.

    Examples
    --------
    Assuming ``model`` is a trained (on the proper-training set) GPyTorch model, 
    and ``cal_x`` and ``cal_y`` are torch tensors with the calibration set inputs 
    and targets. A ``GPRICPWrapper`` with the absolute residual nonconformity 
    measure  (``cpmode='symmetric'``) can be constructed by:
    
    >>> icp = GPRICPWrapper(model, cal_x, cal_y, cpmode='symmetric')

    >>> icp.calibrate(gamma=2.0)
    >>> PIs = icp.predict(X_test, confs=[0.9, 0.95])
    >>> PIs(0.95)

    References
    ----------
    [1] Harris Papadopoulos. Guaranteed Coverage Prediction Intervals with Gaussian Process Regression.
    *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 2024. 
    DOI: `10.1109/TPAMI.2024.3418214 <https://doi.org/10.1109/TPAMI.2024.3418214>`_.
    (`arXiv version <https://arxiv.org/abs/2310.15641>`_).

    [2] Vladimir Vovk, Alexander Gammerman, Glenn Shafer. *Algorithmic Learning in a Random World*, 2nd Ed.
    Springer, 2023. DOI: `10.1007/978-3-031-06649-8 <https://doi.org/10.1007/978-3-031-06649-8>`_.
    """

    def __init__(self, model: GP, cal_inputs: torch.Tensor, cal_targets: torch.Tensor, 
                 cpmode='symmetric',
                 device: torch.device | str | None = None,
                 batch_size: int | None = None):

        if not isinstance(cal_targets, torch.Tensor):
            raise TypeError("cal_targets must be a torch.Tensor")
            
        if cpmode not in ('symmetric', 'asymmetric', None):
            raise ValueError("cpmode must be 'symmetric', 'asymmetric', or None")

        self.model = model
        self._cpmode = cpmode
        self._dtype = torch.float64

        p0 = next(model.parameters(), None)
        self._device = p0.device if p0 is not None else cal_targets.device
        self._model_dtype  = p0.dtype  if p0 is not None else torch.float32

        if device is not None:
            self._device = _normalize_device(device)

        # Move model (and likelihood) to self._device
        self.model.to(device=self._device)
        if hasattr(self.model, "likelihood") and isinstance(self.model.likelihood, torch.nn.Module):
            self.model.likelihood.to(device=self._device)

        # Move calibration targets to device
        self.cal_targets = cal_targets.to(device=self._device, dtype=self._dtype, non_blocking=True)

        self.cal_means, self.cal_vars = _predict_gp(
            self.model,
            cal_inputs,
            batch_size=batch_size,
            device=self._device,
            dtype=self._model_dtype
            )

        self.calibratedICR = None
        self.test_means = None
        self.test_vars  = None

        self._model_fp = self._fingerprint_gp(self.model)

    def refresh_model(self, model: GP, cal_inputs: torch.Tensor, cal_targets: torch.Tensor, 
                 device: torch.device | str | None = None,
                 batch_size: int | None = None):
        """
        Replace the underlying GP model and recompute calibration predictions.

        Parameters
        ----------
        model : gpytorch.models.GP
            New trained model. The model's dtype is **not** modified.
        cal_inputs : torch.Tensor of shape (n_cal, n_features) or DataLoader
            Calibration features (tensor or DataLoader yielding tensors).
        cal_targets : torch.Tensor of shape (n_cal,)
            Calibration targets.
        device : torch.device or str, optional
            Device to place the model on. If omitted, keeps current device.
        batch_size : int, optional
            Batch size used when evaluating the calibration predictions from a tensor.
            If omitted, a single batch with all instances is used.

        Returns
        -------
        GPRICPWrapper
            The wrapper itself (for chaining).

        Notes
        -----
        This resets any previously calibrated ICP state. You must call
        :meth:`calibrate` again before requesting intervals.
        
        Examples
        --------
        Assuming ``icp`` is a ``GPRICPWrapper`` the underlying model of which was 
        modified or we would like to replace with a new one, ``new_model`` is the 
        modified/new GPyTorch model (trained on the proper-training set), and 
        ``cal_x`` and ``cal_y`` are torch tensors with the calibration 
        set inputs and targets. The ``icp`` object is refreshed by:
        
        >>> icp = icp.refresh_model(new_model, cal_x, cal_y)
        
        """

        p0 = next(model.parameters(), None)
        self._device = p0.device if p0 is not None else cal_targets.device
        self._model_dtype  = p0.dtype  if p0 is not None else torch.float32

        if device is not None:
            self._device = _normalize_device(device)

        self.model = model.to(device=self._device)
        if hasattr(self.model, "likelihood") and isinstance(self.model.likelihood, torch.nn.Module):
            self.model.likelihood.to(device=self._device)
            
        self.cal_targets = cal_targets.to(device=self._device, dtype=self._dtype, non_blocking=True)
        self.cal_means, self.cal_vars = _predict_gp(
            self.model,
            cal_inputs,
            batch_size=batch_size,
            device=self._device,
            dtype=self._model_dtype
            )
        
        self.calibratedICR = None
        self.test_means = self.test_vars = None
        
        self._model_fp = self._fingerprint_gp(self.model)
        
        return self

    @property
    def cpmode(self):
        """Get the current Conformal Prediction mode."""
        return self._cpmode

    @cpmode.setter
    def cpmode(self, value):
        """Set the Conformal Prediction mode, ensuring it is one of the acceptable values."""
        if value not in ['symmetric', 'asymmetric', None]:
            raise ValueError("cpmode must be 'symmetric', 'asymmetric', or None")
        self._cpmode = value

    def calibrate(self, gamma=2.0):
        """
        Calibrate the ICP regressor on the stored calibration set.

        Calibration uses the wrapper’s current ``cpmode`` (not passed as an
        argument) and the provided ``gamma``. If ``cpmode`` or ``gamma`` changes,
        call this method again before requesting prediction intervals.

        Parameters
        ----------
        gamma : float, optional, default=2.0
            Power parameter for normalized ICP  nonconformity. If ``gamma >= 1e8``, 
            normalization is short-circuited (equivalent to unnormalized nonconformity).

        Returns
        -------
        GPRICPWrapper
            The wrapper itself (for chaining).

        Notes
        -----
        - If ``cpmode=None`` (CP disabled), this method has no effect.
        - If the underlying model is changed after construction, the wrapper should 
          be updated by calling :meth:`refresh_model` before calibration.
        
        Examples
        --------
        Assuming ``icp`` is a ``GPRICPWrapper``, it can be calibrated with the 
        nonconformity measure parameter (``gamma``) set to 2 by:
    
        >>> icp.calibrate(gamma=2.0)

        """
        
        self._assert_model_unchanged()

        if self.cpmode is None:
            warnings.warn("Calibration was not performed since cpmode is None", RuntimeWarning)
        else:
            self.calibratedICR = InductiveConformalRegressor(
                self.cal_targets, self.cal_means, self.cal_vars, gamma, cpmode=self.cpmode
                )

        return self

    def predict(self, test_inputs=None, confs=None, batch_size: int | None = None):
        """
        Predict on test inputs and produce conformal prediction intervals.

        Parameters
        ----------
        test_inputs : torch.Tensor of shape (n_test, n_features) or torch.utils.data.DataLoader, optional
            Test inputs. If ``None``, reuse cached predictions from a previous call. 
            If a tensor is provided and ``batch_size`` is set, predictions are computed 
            in batches.
        confs : array-like of float in (0, 1), optional, default=[0.95]
            Confidence levels. Ignored when ``cpmode=None``.
        batch_size : int, optional
            Batch size when evaluating a tensor of test inputs.
            If omitted, a single batch with all instances is used.

        Returns
        -------
        PredictionIntervals : gpyconform.PredictionIntervals
            If CP is enabled (``cpmode`` not ``None``) and the wrapper has been
            calibrated, returns prediction intervals for each confidence level in ``confs``. 
            The returned object keeps the intervals on the same device as this
            ``GPRICPWrapper`` instance.
        (means, vars) : tuple[torch.Tensor, torch.Tensor]
            If CP is disabled (``cpmode=None``), returns predictive means and variances.

        Notes
        -----
        - ``__call__`` is an alias of :meth:`predict`.
        - If the underlying model is changed after construction, the wrapper should 
          be updated by calling :meth:`refresh_model` and then calibrated by calling 
          :meth:`calibrate` before prediction.

        Examples
        --------
        Assuming ``icp`` is a calibrated ``GPRICPWrapper`` and ``test_x`` is a torch 
        tensor with the test set inputs, the Conformal Prediction Intervals at the
        90% and 95% confidence levels can be obtained as an instance of 
        ``PredictionIntervals`` by:
    
        >>> PIs = icp.predict(test_x, confs=[0.9, 0.95])
        
        or by:
        
        >>> PIs = icp(test_x, confs=[0.9, 0.95])

        """
        
        self._assert_model_unchanged()

        if test_inputs is None:
            if self.test_means is None:
                raise ValueError("No test_inputs provided and no cached predictions to reuse.")
            warnings.warn(
                "No test_inputs provided; reusing cached test predictions from the previous call.",
                RuntimeWarning
            )
        else:
            self.test_means, self.test_vars = _predict_gp(
                self.model, test_inputs, batch_size=batch_size, 
                device=self._device, dtype=self._model_dtype)

        if self.cpmode is None:
            return self.test_means, self.test_vars
        
        if self.calibratedICR is None:
            raise RuntimeError("Prediction needs a calibrated ICP model. Please call calibrate(...) first.")
        elif self.cpmode != self.calibratedICR.cpmode:
            raise RuntimeError("Current cpmode differs from calibrated model cpmode. Please recalibrate.")

        if confs is None:
            confs = torch.tensor([0.95], device=self._device, dtype=self._dtype)
        else:
            if not isinstance(confs, (torch.Tensor, np.ndarray, list, tuple)):
                raise TypeError(
                    f"'confs' must be a torch.Tensor, numpy.ndarray, list, or tuple, "
                    f"but got {type(confs).__name__}"
                    )

            confs = torch.as_tensor(confs, device=self._device, dtype=self._dtype)

        return self.calibratedICR(self.test_means, self.test_vars, confs)

    def _fingerprint_module(self, mod: torch.nn.Module) -> str:
        import struct, hashlib
        h = hashlib.sha1()
        for name, t in mod.state_dict().items():
            h.update(name.encode())
            h.update(str(t.dtype).encode())
            # pack rank then dims as 64-bit ints
            h.update(struct.pack("<Q", len(t.shape)))
            for d in t.shape:
                h.update(struct.pack("<Q", int(d)))
            if t.numel():
                h.update(t.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

    def _fingerprint_gp(self, model) -> str:
        h = hashlib.sha1(self._fingerprint_module(model).encode())
        if hasattr(model, "likelihood") and isinstance(model.likelihood, torch.nn.Module):
            h.update(self._fingerprint_module(model.likelihood).encode())
        return h.hexdigest()

    def _assert_model_unchanged(self):
        current = self._fingerprint_gp(self.model)
        if current != self._model_fp:
            raise RuntimeError("Underlying model changed. Please call refresh_model(...).")

    __call__ = predict
