r"""
Initialize the gpyconform package, applying patches for adding the Conformal prediction strategies.
"""

__version__ = '0.1.0'

from .exact_prediction_strategies_cp import apply_patches
apply_patches()

from .exact_gpcp import ExactGPCP
from .prediction_intervals import PredictionIntervals

__all__ = ['__version__', 'ExactGPCP', 'PredictionIntervals']