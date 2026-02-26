#!/usr/bin/env python3

import torch
import warnings

from gpyconform.constants import _CONF_DP, _CONF_SCALE

class PredictionIntervals:
    """
    Contains the conformal Prediction Intervals (PIs) at one or more confidence levels 
    and provides functionality for their retrieval and evaluation.
    
    Notes
    -----
    Confidence levels are internally stored using fixed-point integer keys to avoid
    float equality issues when retrieving intervals for a requested level.

    """

    def __init__(self, conf_levels: torch.Tensor, all_pis: torch.Tensor):
        self.all_pis = all_pis
        
        # Fixed-point integer keys: exact, to avoid float-equality issues
        conf_scaled = torch.round(conf_levels * _CONF_SCALE)
        self._conf_keys = conf_scaled.to(torch.int64)
        
        # One-time index map: int_key -> row index in all_pis
        self._index = {int(k.item()): i for i, k in enumerate(self._conf_keys)}
    
    def __call__(self, conf_level: float | None = None, *, y_min = float('-inf'), y_max = float('inf'), dp: int = 6):
        """
        Returns the Prediction Intervals for a specified confidence level or all intervals if 
        confidence level is not specified.
        
        Parameters
        ----------
        conf_level : float in range (0,1), optional
            Confidence level for which to return the corresponding Prediction Intervals.
            If not specified, the Prediction Intervals for all confidence levels will be returned.
        y_min : float, keyword-only, default=-inf
            If provided, PIs are cut to exclude values below y_min.
        y_max : float, keyword-only, default=inf
            If provided, PIs are cut to exclude values above y_max.        
        dp : int in range [1,6], keyword-only, default=6
            Number of decimals to show in the string keys when returning all levels.

        Returns
        -------
        torch.Tensor or dict[str, torch.Tensor]
            A torch tensor with the Prediction Intervals for the specified ``conf_level``, or a dictionary 
            with confidence levels as keys (str) and the corresponding Prediction Interval 
            tensors as values if ``conf_level`` is None.

        Examples
        --------
        Assuming ``PIs`` is an instance of ``PredictionIntervals`` that includes the 95% 
        confidence level.

        To retrieve the Prediction Intervals at the 95% confidence level as a tensor:

        >>> intervals = PIs(0.95)
        >>> print(intervals)

        To retrieve the Prediction Intervals for all confidence levels as a dictionary:

        >>> all_intervals = PIs()
        >>> print(all_intervals)
        
        """

        if not isinstance(dp, int):
            raise TypeError(f"dp must be an int, got {type(dp).__name__}.")
        if not (1 <= dp <= _CONF_DP):
            raise ValueError(f"dp must be an integer in [1, {_CONF_DP}], got {dp}.")
            
        if conf_level is None:
            # Create a dictionary of all prediction intervals (string keys -> interval tensors)
            out = {}
            for i, cl in enumerate(self._conf_keys):
                cl_str = f"{(cl.item() / _CONF_SCALE):.{dp}f}"
                cl_pis = self.all_pis[i].clone()
                cl_pis[:, 0] = cl_pis[:, 0].clamp(min=y_min)
                cl_pis[:, 1] = cl_pis[:, 1].clamp(max=y_max)        
                out[cl_str] = cl_pis
            return out

        cl = int(round(float(conf_level) * _CONF_SCALE))
        idx = self._index.get(cl)
        if idx is None:
            available = ", ".join(
                self._format_level_key(cl) for cl in self._index.keys()
            )
            raise ValueError(
                f"Confidence level {conf_level} not found. Available levels are: [{available}]"
            )
            
        out = self.all_pis[idx].clone()
        out[:, 0] = out[:, 0].clamp(min=y_min)
        out[:, 1] = out[:, 1].clamp(max=y_max)

        return out

        
    def evaluate(self, conf_level, metrics=None, y=None, *, y_min = float('-inf'), y_max = float('inf')):
        """
        Evaluates the Prediction Intervals at a specified confidence level.

        Parameters
        ----------
        conf_level : float in range (0,1)
            Confidence level of the Prediction Intervals to be evaluated.
        metrics : list of str or str, optional, default=['mean_width', 'median_width', 'error']
            Metrics to calculate. Possible options:
            - 'mean_width': Average width of the Prediction Intervals.
            - 'median_width': Median width of the Prediction Intervals.
            - 'error': Percentage of Prediction Intervals that do not contain the true target value.
        y : torch.Tensor of shape (n_test,), optional, default=None
            True target values, required for calculating the 'error' metric. If not provided, 'error' 
            is not calculated.
        y_min : float, keyword-only, default=-inf
            If provided, PIs are evaluated after cutting values below y_min.
        y_max : float, keyword-only, default=inf
            If provided, PIs are evaluated after cutting values above y_max.        

        Returns
        -------
        results : dict
            A dictionary with a key for each metric in ``metrics`` and the 
            calculated result as its value. For example: {'mean_width': 3.852, 'error': 0.049}.

        Examples
        --------
        Assuming ``PIs`` is an instance of ``PredictionIntervals`` that includes the 99% 
        confidence level, and ``test_y`` is a tensor with the true targets.

        To evaluate the Prediction Intervals at the 99% confidence level using all available metrics 
        (which is the default):

        >>> results = PIs.evaluate(0.99, y=test_y)
            
        To evaluate only the mean width of the Prediction Intervals at the 99% confidence level:

        >>> results = PIs.evaluate(0.99, metrics='mean_width')
        
        """

        if metrics is None:
            metrics = ['mean_width', 'median_width', 'error']
        
        if isinstance(metrics, str):
            metrics = [metrics]

        cl_pis = self(conf_level, y_min=y_min, y_max=y_max)

        results = {}

        # Check if any metrics require pi_widths before calculating
        need_widths = any(m in ('mean_width', 'median_width') for m in metrics)
        if need_widths:
            pi_widths = cl_pis[:, 1] - cl_pis[:, 0]

        for name in metrics:
            if name == 'error':
                if y is None:
                    warnings.warn(
                        "True labels 'y' not provided for error calculation - skipping 'error' metric.",
                        RuntimeWarning
                    )
                else:
                    y = torch.as_tensor(y, device=cl_pis.device, dtype=cl_pis.dtype)
                    errors = (y < cl_pis[:, 0]) | (y > cl_pis[:, 1])
                    results['error'] = errors.to(torch.float64).mean().item()
            elif name == 'mean_width':
                results['mean_width'] = pi_widths.mean().item()
            elif name == 'median_width':
                results['median_width'] = pi_widths.median().item()
            else:
                warnings.warn(f"'{name}' is not a recognized metric.", RuntimeWarning)

        return results


    def _decimal_places_for_key(self, int_key: int) -> int:
        tz = 0
        tmp = abs(int_key)
        # count trailing zeros, up to the stored precision
        while tz < _CONF_DP-2 and (tmp % 10) == 0:
            tz += 1
            tmp //= 10
        
        return _CONF_DP - tz

    def _format_level_key(self, int_key: int) -> str:
        dp = self._decimal_places_for_key(int_key)

        return f"{int_key / _CONF_SCALE:.{dp}f}"