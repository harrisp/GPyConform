# Changelog

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
