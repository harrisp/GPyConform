# Changelog

## v0.2.0.post1 (01/03/2026)

### Fixes
- Documentation build/configuration fixes.
- Packaging metadata updates (no functional code changes).


## v0.2.0 (26/02/2026)

### Features
- **Inductive (Split) Conformal Prediction (ICP) for GPR**: Added ICP support with `GPRICPWrapper` (GPyTorch model wrapper) and a model-agnostic `InductiveConformalRegressor`.
- **Tensor split utilities**: Added `tensor_train_test_split` and `tensor_train_cal_split` to conveniently create train/test and proper-train/calibration splits directly on `torch.Tensor`s.

### Updates
- **Configurable patching via environment variable**: Added `GPYCONFORM_AUTOPATCH` to control when/if the internal GPyTorch prediction-strategy patch is applied (eager, lazy, or forbidden). Exposed `apply_patches()` / `is_patched()` for manual control.

### Fixes
- **Precision / numerical stability improvements**: Improved numerical robustness in interval construction and confidence-level handling.


## v0.1.1 (09/12/2024)

### Updates
- **GPyTorch Compatibility**: Updated the required GPyTorch version to v1.13.

### Fixes
- **Precision Issue in PredictionIntervals**: Resolved a floating-point precision issue that prevented some confidence levels from being correctly identified.


## v0.1.0 (21/10/2024) - Initial Release

🎉 Welcome to the first release of [GPyConform](https://github.com/harrisp/GPyConform)! **GPyConform** extends the [GPyTorch](https://gpytorch.ai) library by implementing (Full) Conformal Prediction for Gaussian Process Regression.

### Features
- **Provably Valid Prediction Intervals**: Provides Prediction Intervals with guaranteed coverage under minimal assumptions (data exchangeability).
- **Full Utilization of GPyTorch**: Leverages the robust and efficient GP modeling capabilities of GPyTorch.
- **Supports Both Symmetric and Asymmetric Prediction Intervals**: Implements both the symmetric and asymmetric Full Conformal Prediction approaches for constructing Prediction Intervals.
