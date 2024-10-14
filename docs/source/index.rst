.. GPyConform documentation master file, created by
   sphinx-quickstart on Sat Oct  5 17:22:41 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

GPyConform's documentation
======================================

GPyConform extends the GPyTorch library to implement Gaussian Process Regression Conformal Prediction based on the approach described in [1]. 
Designed to work seamlessly with Exact Gaussian Process (GP) models, GPyConform enhances GPyTorch by introducing the capability to generate 
and evaluate both 'symmetric' and 'asymmetric' Conformal Prediction Intervals.

Key Features
------------
- **Provides Provably Valid Prediction Intervals**: Provides Prediction Intervals with guaranteed coverage under minimal assumptions (data exchangeability).
- **Inherits All GPyTorch Functionality**: Utilizes the robust and efficient GP modeling capabilities of GPyTorch.
- **Supports Both Symmetric and Asymmetric Prediction Intervals**: Implements both Full Conformal Prediction approaches for constructing Prediction Intervals.

.. note::

   This documentation focuses primarily on the additional functionality provided by GPyConform. For detailed information on the usage and features of the 
   underlying GPyTorch functionality, please refer to the `GPyTorch documentation <https://gpytorch.readthedocs.io/en/latest/>`_.

.. note::

   Currently, GPyConform is tailored specifically for GPyTorch's ExactGP models combined with any mean function and any covariance function that employs an 
   exact prediction strategy.

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
