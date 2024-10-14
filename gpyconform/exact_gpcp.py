#!/usr/bin/env python3

import warnings
import torch

from gpytorch import settings
from gpytorch.distributions import MultivariateNormal
from gpytorch.utils.generic import length_safe_zip
from gpytorch.utils.warnings import GPInputWarning
from gpytorch.models import ExactGP
from gpytorch.models.exact_prediction_strategies import prediction_strategy

class ExactGPCP(ExactGP):
    r"""
    Extends GPyTorch's ExactGP to produce Conformal Prediction Intervals, specifically modifying behavior 
    only in the evaluation (``.eval()``) mode. In particular, it implements both the symmetric approach described 
    in [1] and its asymmetric version, following the approach described in Chapter 2.3 of [2].
    For more details on the inherited functionality of ExactGP please see GPyTorch's documentation 
    at: `GPyTorch Docs <https://gpytorch.readthedocs.io/en/latest/>`_.

    Parameters
    ----------
    train_inputs : torch.Tensor of shape (n, d), denoted as :math:`\mathbf{X}`
        Training features.

    train_targets : torch.Tensor of shape (n), denoted as :math:`\mathbf{y}`
        Training targets.

    likelihood : gpytorch.likelihoods.GaussianLikelihood
        The Gaussian likelihood defining the observational distribution, necessary for exact inference.

    cpmode : 'symmetric' or 'asymmetric' or None, default='symmetric' 
        Mode of the Conformal Prediction: 

        - 'symmetric': Employs the absolute residual nonconformity measure approach as described in [1].
        - 'asymmetric': Employs the asymmetric version of the nonconformity measure defined in [1], following the approach described in Chapter 2.3 of [2].
        - None: Reverts to GPyTorch's ExactGP behavior.

    Raises
    ------
    ValueError
        If ``cpmode`` is not 'symmetric', 'asymmetric', or None.

    Note
    ----
    The ``cpmode`` property can change at any time without affecting the model.


    References
    ----------
    [1] Harris Papadopoulos. Guaranteed Coverage Prediction Intervals with Gaussian Process Regression.
    *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 2024. 
    DOI: `10.1109/TPAMI.2024.3418214 <https://doi.org/10.1109/TPAMI.2024.3418214>`_.
    (`arXiv version <https://arxiv.org/abs/2310.15641>`_).

    [2] Vladimir Vovk, Alexander Gammerman, Glenn Shafer. *Algorithmic Learning in a Random World*, 2nd Ed.
    Springer, 2023. DOI: `10.1007/978-3-031-06649-8 <https://doi.org/10.1007/978-3-031-06649-8>`_.


    Example
    -------
    Assuming ``train_x`` and ``train_y`` are torch tensors with the training features and targets respectively,
    a Gaussian Process Regression model with Conformal Prediction capabilities can be formed by:
    
    .. code-block:: python

        # Construct the model
        class MyGPCP(gpyconform.ExactGPCP):
            def __init__(self, train_x, train_y, likelihood, cpmode='symmetric'):
                super(MyGPCP, self).__init__(train_x, train_y, likelihood, cpmode=cpmode)
                self.mean_module = gpytorch.means.ZeroMean()  # Prior mean - any mean module can be used
                self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

            def forward(self, x):
                mean = self.mean_module(x)
                covar = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean, covar)

        # Initialize likelihood and model
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = MyGPCP(train_x, train_y, likelihood)
        
        # If needed change the cpmode property at any time
        model.cpmode = 'asymmetric'

    Note
    ----
    Any mean function from ``gpytorch.means`` and any kernel function that employs an exact prediction strategy
    from ``gpytorch.kernels`` can be used with GPyConform.

    """

    def __init__(self, train_inputs, train_targets, likelihood, cpmode='symmetric'):
        self.cpmode = cpmode
        super().__init__(train_inputs, train_targets, likelihood)

    @property
    def cpmode(self):
        """Get the mode of Conformal Prediction."""
        return self._cpmode

    @cpmode.setter
    def cpmode(self, value):
        """Set the mode of Conformal Prediction, ensuring it is one of the acceptable values."""
        if value not in ['symmetric', 'asymmetric', None]:
            raise ValueError("cpmode must be 'symmetric', 'asymmetric', or None")
        self._cpmode = value
        
    def __call__(self, *args, **kwargs):
        r"""
        In evaluation (``.eval()``) mode, calling this model with test inputs will return the symmetric or 
        asymmetric Conformal Prediction Intervals depending on ``cpmode``. Parameters for ``.eval()`` mode:

        Parameters
        ----------
        test_inputs : torch.Tensor
            Test features.
        gamma : float, default=2
            The gamma parameter of the nonconformity measure, which controls its sensitivity.
        confs : torch.Tensor or numpy.array or list, default=torch.tensor([0.95])
            Confidence levels for which to return Prediction Intervals. Each confidence level must be 
            a float in the range (0,1).
    
        Raises
        ------
        ValueError
            If any confidence level in ``confs`` is not in the range (0,1).

        Returns
        -------
        PIs : gpyconform.PredictionIntervals
            An object containing the Prediction Intervals for each confidence level in ``confs``.

        Note
        ----
        The ``gamma`` and ``confs`` parameters are used only in ``.eval()`` mode. They are ignored in 
        all other cases.

        Example
        -------
        Assuming ``model`` is an instance of a GP Conformal Regressor, with optimized hyperparameters, 
        and ``test_x`` is a torch tensor containing the test features. The Conformal Prediction Intervals 
        at the 90%, 95%, and 99% confidence levels, with the nonconformity measure parameter ``gamma`` 
        set to 2, can be obtained as an instance of ``PredictionIntervals`` by:
    
        .. code-block:: python

            model.eval()
            
            with torch.no_grad():  # Disable gradient calculation
                PIs = model(test_x, gamma=2, confs=[0.9, 0.95, 0.99])
        """

        gamma = kwargs.pop('gamma', 2)
        confs = kwargs.pop('confs', torch.tensor([0.95]))

        if any((conf < 0 or conf > 1) for conf in confs):
            raise ValueError("All confidence levels in 'confs' must be in (0,1)")

        if self.training or settings.prior_mode.on() or self.train_inputs is None or self.train_targets is None or self.cpmode is None:
            return super().__call__(*args, **kwargs)    
        else:
            train_inputs = list(self.train_inputs)
            inputs = [i.unsqueeze(-1) if i.ndimension() == 1 else i for i in args]

            if settings.debug.on():
                if all(torch.equal(train_input, input) for train_input, input in length_safe_zip(train_inputs, inputs)):
                    warnings.warn(
                        "The input matches the stored training data. Did you forget to call model.train()?",
                        GPInputWarning,
                    )

            if self.prediction_strategy is None:
                train_output = super(ExactGP,self).__call__(*train_inputs, **kwargs)

                # Create the prediction strategy
                self.prediction_strategy = prediction_strategy(
                    train_inputs=train_inputs,
                    train_prior_dist=train_output,
                    train_labels=self.train_targets,
                    likelihood=self.likelihood,
                )

            # Concatenate the input to the training input
            full_inputs = []
            batch_shape = train_inputs[0].shape[:-2]
            for train_input, input in length_safe_zip(train_inputs, inputs):
                # Make sure the batch shapes agree for training/test data
                if batch_shape != train_input.shape[:-2]:
                    batch_shape = torch.broadcast_shapes(batch_shape, train_input.shape[:-2])
                    train_input = train_input.expand(*batch_shape, *train_input.shape[-2:])
                if batch_shape != input.shape[:-2]:
                    batch_shape = torch.broadcast_shapes(batch_shape, input.shape[:-2])
                    train_input = train_input.expand(*batch_shape, *train_input.shape[-2:])
                    input = input.expand(*batch_shape, *input.shape[-2:])
                full_inputs.append(torch.cat([train_input, input], dim=-2))

            # Get the joint distribution for training/test data
            full_output = super(ExactGP, self).__call__(*full_inputs, **kwargs)
            if settings.debug().on():
                if not isinstance(full_output, MultivariateNormal):
                    raise RuntimeError("ExactGP.forward must return a MultivariateNormal")
            full_mean, full_covar = full_output.loc, full_output.lazy_covariance_matrix

            # Make the prediction -> PIs
            with settings.cg_tolerance(settings.eval_cg_tolerance.value()):
                out = self.prediction_strategy.exact_prediction(full_mean, full_covar, gamma=gamma, confs=confs, cpmode=self.cpmode)

            return out
