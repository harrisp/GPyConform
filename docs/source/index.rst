.. GPyConform documentation master file, created by
   sphinx-quickstart on Sat Oct  5 17:22:41 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

GPyConform's documentation
======================================

**GPyConform** extends the `GPyTorch <https://gpytorch.ai>`_ library by implementing **Conformal Prediction (CP) for Gaussian Process Regression (GPR)**, providing **distribution-free, finite-sample valid Prediction Intervals (PIs)** under the sole assumption of data exchangeability.

GPyConform supports both the **Transductive (Full) CP** and **Inductive (Split) CP** versions of the framework through a unified interface. In both versions it implements a **GPR-specific normalized nonconformity measure** [1] that leverages the GP predictive variance to construct adaptive *symmetric* or *asymmetric* conformal prediction intervals.

Key Features
------------
- **Provably Valid Prediction Intervals**: Distribution-free, finite-sample coverage guarantees under minimal assumptions (data exchangeability).
- **Two CP Framework Versions**:
  - **Transductive (Full) CP** for Exact GPs: ``ExactGPCP``
  - **Inductive (Split) CP** for any GPyTorch regression model: ``GPRICPWrapper``, plus a model-agnostic ``InductiveConformalRegressor``
- **Symmetric and Asymmetric PIs** in both frameworks.
- **Normalized Nonconformity** that leverages the GP predictive variance for tighter, adaptive intervals.
- **Unified PI Container + Metrics**: ``PredictionIntervals`` supports retrieving intervals at multiple confidence levels and evaluating empirical coverage error and interval widths.
- **Torch-native + GPU-friendly**: Works directly with ``torch.Tensor``\s and can leverage GPU acceleration.

.. note::

   This documentation focuses primarily on the additional functionality provided by GPyConform. For detailed information on the usage and features of the
   underlying GPyTorch functionality, please refer to the `GPyTorch documentation <https://gpytorch.readthedocs.io/en/latest/>`_.

.. note::

   - **Transductive CP** (``ExactGPCP``) targets **ExactGP models with ``GaussianLikelihood``** and relies on an internal patch to GPyTorch’s ``DefaultPredictionStrategy`` (applied automatically by default). You can control patching via the ``GPYCONFORM_AUTOPATCH`` environment variable, or call ``gpyconform.apply_patches()`` manually.
   - **Inductive CP** does **not** modify model internals and can be used with **any** GPyTorch regression model (including approximate/deep GPs and different likelihoods). ``InductiveConformalRegressor`` can also be used with non-GPyTorch regressors that provide predictive means/variances.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   Getting started <getting_started.md>
   The GPyConform package <gpyconform.rst>	      
   Citing GPyConform <citing.md>
   References <references.md>

..
   Indices and tables
   ==================
   
   * :ref:`genindex`
   * :ref:`modindex`
   * :ref:`search`
