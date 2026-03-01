# GPyConform

[![Python Version](https://img.shields.io/badge/Python-3.10+-orange.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![PyPI Version](https://img.shields.io/pypi/v/gpyconform.svg)](https://pypi.org/project/gpyconform)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/gpyconform.svg)](https://anaconda.org/conda-forge/gpyconform)
[![GitHub (Pre-)Release Date](https://img.shields.io/github/release-date-pre/harrisp/gpyconform)](https://github.com/harrisp/gpyconform/blob/master/CHANGELOG.md)
[![Documentation Status](https://readthedocs.org/projects/gpyconform/badge/?version=latest)](https://gpyconform.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](https://github.com/harrisp/gpyconform/blob/master/LICENSE.txt)
[![Downloads](https://static.pepy.tech/badge/gpyconform)](https://pepy.tech/project/gpyconform)


**GPyConform** extends the [GPyTorch](https://gpytorch.ai) library by implementing **Conformal Prediction (CP) for Gaussian Process Regression (GPR)**, providing **distribution-free, finite-sample valid Prediction Intervals (PIs)** under the sole assumption of data exchangeability.

GPyConform supports both the **Transductive (Full) CP** and **Inductive (Split) CP** versions of the framework through a unified interface. In both cases it implements a **GPR-specific normalized nonconformity measure** [1] that leverages the GP predictive variance to construct adaptive *symmetric* or *asymmetric* conformal prediction intervals.

## Key Features
- **Provably Valid Prediction Intervals**: Distribution-free, finite-sample coverage guarantees under minimal assumptions (data exchangeability).
- **Two CP Framework Versions**:
  - **Transductive (Full) CP** for Exact GPs: `ExactGPCP`
  - **Inductive (Split) CP** for any GPyTorch regression model: `GPRICPWrapper`, plus a model-agnostic `InductiveConformalRegressor`
- **Symmetric and Asymmetric PIs** in both frameworks.
- **Normalized Nonconformity** that leverages the GP predictive variance for tighter, adaptive intervals.
- **Unified PI Container + Metrics**: `PredictionIntervals` supports retrieving intervals at multiple confidence levels and evaluating empirical coverage error and interval widths.
- **Torch-native + GPU-friendly**: Works directly with `torch.Tensor`s and can leverage GPU acceleration.

### Note
- **Transductive CP** (`ExactGPCP`) targets **ExactGP models with `GaussianLikelihood`** and relies on an internal patch to GPyTorch’s `DefaultPredictionStrategy` (applied automatically by default). You can control patching via the `GPYCONFORM_AUTOPATCH` environment variable, or call `gpyconform.apply_patches()` manually.
- **Inductive CP** does **not** modify the model internals and can be used with **any** GPyTorch regression model (including approximate/deep GPs and different likelihoods). `InductiveConformalRegressor` can also be used with non-GPyTorch regressors that provide predictive means/variances. 

## Documentation

For detailed documentation and usage examples, see [GPyConform Documentation](https://gpyconform.readthedocs.io).

## Installation

From [PyPI](https://pypi.org/project/gpyconform/)

```bash
pip install gpyconform
```

From [conda-forge](https://anaconda.org/conda-forge/gpyconform)

```bash
conda install conda-forge::gpyconform
```

## Citing GPyConform

If you use `GPyConform` for a scientific publication, you are kindly requested to cite the following paper:

Harris Papadopoulos. "GPyConform: Conformal Prediction with Gaussian Process Regression in Python". In: K. An Nguyen, Z. Luo (eds), *The Importance of Being Learnable*, Lecture Notes in Computer Science, vol. 16290, pp. 449–466. Springer, 2026. DOI: [10.1007/978-3-032-15120-9_20](https://doi.org/10.1007/978-3-032-15120-9_20).

Bibtex entry:

```bibtex
@Inbook{gpyconform,
author="Papadopoulos, Harris",
editor="An Nguyen, Khuong and Luo, Zhiyuan",
title="GPyConform: Conformal Prediction with Gaussian Process Regression in Python",
bookTitle="The Importance of Being Learnable: Essays Dedicated to Alexander Gammerman",
year="2026",
publisher="Springer Nature Switzerland",
address="Cham",
pages="449--466",
isbn="978-3-032-15120-9",
doi="10.1007/978-3-032-15120-9_20",
url="https://doi.org/10.1007/978-3-032-15120-9_20"
}
```

For the Gaussian Process Regression Conformal Prediction approach and nonconformity measure, please also cite:

Harris Papadopoulos. "Guaranteed Coverage Prediction Intervals with Gaussian Process Regression", *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 46, no. 12, pp. 9072-9083, Dec. 2024. DOI: [10.1109/TPAMI.2024.3418214](https://doi.org/10.1109/TPAMI.2024.3418214).
([arXiv version](https://arxiv.org/abs/2310.15641))

Bibtex entry:

```bibtex
@ARTICLE{gprcp,
  author={Papadopoulos, Harris},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Guaranteed Coverage Prediction Intervals with Gaussian Process Regression}, 
  year={2024},
  volume={46},
  number={12},
  pages={9072-9083},
  doi={10.1109/TPAMI.2024.3418214}
}
```

## References

<a id="1">[1]</a> Harris Papadopoulos. "Guaranteed Coverage Prediction Intervals with Gaussian Process Regression", *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 46, no. 12, pp. 9072-9083, Dec. 2024. DOI: [10.1109/TPAMI.2024.3418214](https://doi.org/10.1109/TPAMI.2024.3418214). 
([arXiv version](https://arxiv.org/abs/2310.15641))

<a id="2">[2]</a> Vladimir Vovk, Alexander Gammerman, and Glenn Shafer. *Algorithmic Learning in a Random World*, 2nd Ed. Springer, 2023. DOI: [10.1007/978-3-031-06649-8](https://doi.org/10.1007/978-3-031-06649-8).

<a id="3">[3]</a> Harris Papadopoulos. "GPyConform: Conformal Prediction with Gaussian Process Regression in Python". In: K. An Nguyen, Z. Luo (eds), *The Importance of Being Learnable*, Lecture Notes in Computer Science, vol. 16290, pp. 449–466. Springer, 2026. DOI: [10.1007/978-3-032-15120-9_20](https://doi.org/10.1007/978-3-032-15120-9_20).

- - -

Author: Harris Papadopoulos (h.papadopoulos@frederick.ac.cy) / 
Copyright 2024-2026 Harris Papadopoulos / 
License: BSD 3 clause
