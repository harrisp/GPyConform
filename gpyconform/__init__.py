"""
Initialize the gpyconform package and (optionally) apply the gpyconform 
prediction-strategy patch required for transductive conformal prediction.

Environment variables
---------------------
GPYCONFORM_AUTOPATCH
    Controls when/if the patch is applied:

    - "1": apply the patch eagerly at import time.
    - "0": forbid patching (constructing :class:`gpyconform.ExactGPCP` will raise).
    - unset/other: apply the patch lazily when :class:`gpyconform.ExactGPCP` is
      instantiated, or call :func:`gpyconform.apply_patches` manually.
"""

__version__ = '0.2.0'

import os
from .exact_prediction_strategies_cp import apply_patches, is_patched

if os.getenv("GPYCONFORM_AUTOPATCH", "") == "1":
    apply_patches()

from .exact_gpcp import ExactGPCP
from .gpricp import GPRICPWrapper, InductiveConformalRegressor
from .prediction_intervals import PredictionIntervals

__all__ = ['__version__', 
           'ExactGPCP', 
           'GPRICPWrapper', 
           'InductiveConformalRegressor', 
           'PredictionIntervals',
           'apply_patches',
           'is_patched']