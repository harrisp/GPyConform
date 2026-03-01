The GPyConform Package
======================

.. module:: gpyconform
.. currentmodule:: gpyconform

Transductive (Full) Conformal Prediction
----------------------------------------

.. autoclass:: ExactGPCP
   :special-members: __call__
   :exclude-members: __init__, cpmode

Inductive (Split) Conformal Prediction
--------------------------------------

.. autoclass:: InductiveConformalRegressor
   :special-members: __call__

.. autoclass:: GPRICPWrapper
   :members: calibrate, predict, refresh_model

Conformal Prediction Intervals
------------------------------

.. autoclass:: PredictionIntervals
   :members: evaluate
   :special-members: __call__
