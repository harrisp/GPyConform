#!/usr/bin/env python3

import torch
import warnings

class PredictionIntervals:
    r"""
    Contains the Conformal Prediction Intervals and provides functionality for their retrieval 
    and evaluation. 
    """
    
    def __init__(self, conf_levels, all_pis):
        self.conf_levels = conf_levels
        self.all_pis = all_pis
    
    def __call__(self, conf_level=None):
        r"""
        Returns the Prediction Intervals for a specified confidence level or all intervals if 
        confidence level is not specified.
        
        Parameters
        ----------
        conf_level : float in range (0,1), optional
            Confidence level for which to return the corresponding Prediction Intervals.
            If not specified, the Prediction Intervals for all confidence levels will be returned.

        Raises
        ------
        ValueError
            If ``conf_level`` is specified but not among the levels available in the particular instance.

        Returns
        -------
        torch.Tensor or dict[float, torch.Tensor]
            A torch tensor with the Prediction Intervals for the specified ``conf_level``, or a dictionary 
            with confidence levels as keys and the corresponding Prediction Interval tensors as values 
            if ``conf_level`` is None.

        Example
        -------
        Assuming ``PIs`` is an instance of ``PredictionIntervals`` that includes the 95% 
        confidence level.

        To retrieve the Prediction Intervals at the 95% confidence level as a tensor:

        .. code-block:: python

            intervals = PIs(0.95)
            print(intervals)

        To retrieve the Prediction Intervals for all confidence levels as a dictionary:

        .. code-block:: python

            all_intervals = PIs()
            print(all_intervals)
        """

        if conf_level is None:
            # Create a dictionary of all prediction intervals, using confidence levels as keys
            return {round(float(cl), 5): self.all_pis[i, :, :] for i, cl in enumerate(self.conf_levels)}

        indices = torch.nonzero(self.conf_levels == conf_level, as_tuple=True)[0]
        if indices.numel() > 0:
            return self.all_pis[indices[0], :, :]
        else:
            raise ValueError(f"Confidence level {conf_level} not found. Available levels are: {self.conf_levels.numpy()}")
        
    def evaluate(self, conf_level, metrics=['mean_width', 'median_width', 'error'], y=None):
        r"""
        Evaluates the Prediction Intervals for a specified confidence level.

        Parameters
        ----------
        conf_level : float in range (0,1)
            Confidence level of the Prediction Intervals to be evaluated.
        metrics : list of str or str, default=['mean_width', 'median_width', 'error']
            Metrics to calculate. Possible options:
            
            - 'mean_width': Average width of the Prediction Intervals.
            - 'median_width': Median width of the Prediction Intervals.
            - 'error': Percentage of Prediction Intervals that do not contain the true target value.
        y : torch.Tensor, default=None
            True target values, required for calculating the 'error' metric. If not provided, 'error' 
            is not calculated.

        Raises
        ------
        ValueError
            If ``conf_level`` is not among the levels available in the particular instance.
        RuntimeWarning
            If ``error`` is in ``metrics`` but ``y`` is None; error calculation requires true target values. 
        RuntimeWarning
            If unrecognized metrics are specified in ``metrics``; these are ignored.

        Returns
        -------
        results : dict
            A dictionary with a key for each metric in ``metrics`` and the 
            calculated result as its value. For example: {'mean_width': 3.852, 'error': 0.049}.

        Example
        -------
        Assuming ``PIs`` is an instance of ``PredictionIntervals`` that includes the 99% 
        confidence level, and ``test_y`` is a tensor with the true targets.

        To evaluate the Prediction Intervals at the 99% confidence level using all available metrics 
        (which is the default):

        .. code-block:: python

            results = PIs.evaluate(0.99, y=test_y)
            
        To evaluate only the mean width of the Prediction Intervals at the 99% confidence level:

        .. code-block:: python

            results = PIs.evaluate(0.99, metrics='mean_width')
        """
        
        unobserved_functions = {
             'mean_width': self._mean_pi_width,
             'median_width': self._median_pi_width
        }
    
        if isinstance(metrics, str):
            metrics = [metrics]
    
        results = {}

        indices = torch.nonzero(self.conf_levels == conf_level, as_tuple=True)[0]
        if indices.numel() > 0:
            conf_index = indices[0]
        else:
            raise ValueError(f"Confidence level {conf_level} not found. Available levels are: {self.conf_levels.numpy()}")
        
        # Check if any metrics require pi_widths before calculating
        pi_widths_required = any(metric in unobserved_functions for metric in metrics)
        if pi_widths_required:
            pi_widths = self.all_pis[conf_index,:,1] - self.all_pis[conf_index,:,0]

        for name in metrics:
            if name == 'error':
                if y is None:
                    warnings.warn("True labels 'y' not provided for error calculation - skipping 'error' metric.", RuntimeWarning)
                else:
                    results['error'] = self._error_percentage(conf_index, y)
            elif name in unobserved_functions:
                result = unobserved_functions[name](conf_index, pi_widths)
                results[name] = result
            else:
                warnings.warn(f"'{name}' is not a recognized metric.", RuntimeWarning)
    
        return results

    def _error_percentage(self, conf_index, y):
        r"""
        Returns the percentage of errors for a given confidence level.

        Parameters
        ----------
        conf_index : int
            index of the confidence level in conf_levels
        y : torch.Tensor
            True target values

        Returns
        -------
        float
            Percentage of errors (true target value not in Prediction Interval)
        """
        errors = (y < self.all_pis[conf_index,:,0]) | (y > self.all_pis[conf_index,:,1])

        num_errors = torch.sum(errors, dtype=torch.float32)
        total = errors.numel()
        prc_errors = num_errors / total

        return prc_errors.item()

    def _mean_pi_width(self, conf_index, pi_widths):
        return pi_widths.mean().item()
    
    def _median_pi_width(self, conf_index, pi_widths):
        return pi_widths.median().item()
        