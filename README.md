# GPyConform

[![Python Version](https://img.shields.io/badge/Python-3.8+-orange.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![PyPI Version](https://img.shields.io/pypi/v/gpyconform.svg)](https://pypi.org/project/gpyconform)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/gpyconform.svg)](https://anaconda.org/conda-forge/gpyconform)
[![GitHub (Pre-)Release Date](https://img.shields.io/github/release-date-pre/harrisp/gpyconform)](https://github.com/harrisp/gpyconform/blob/master/CHANGELOG.md)
[![Documentation Status](https://readthedocs.org/projects/gpyconform/badge/?version=latest)](https://gpyconform.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](https://github.com/harrisp/gpyconform/blob/master/LICENSE.txt)
[![Downloads](https://static.pepy.tech/badge/gpyconform)](https://pepy.tech/project/gpyconform)


**GPyConform** extends the [GPyTorch](https://gpytorch.ai) library by implementing (Full) Conformal Prediction for Gaussian Process Regression based on the approach described in [1]. Designed to work seamlessly with Exact Gaussian Process (GP) models, GPyConform enhances GPyTorch by introducing the capability to generate and evaluate both 'symmetric' and 'asymmetric' Conformal Prediction Intervals.

## Key Features
- **Provides Provably Valid Prediction Intervals**: Provides Prediction Intervals with guaranteed coverage under minimal assumptions (data exchangeability).
- **Full Utilization of GPyTorch**: Leverages the robust and efficient GP modeling capabilities of GPyTorch.
- **Supports Both Symmetric and Asymmetric Prediction Intervals**: Implements both the symmetric and asymmetric Full Conformal Prediction approaches for constructing Prediction Intervals.

### Note
Currently, GPyConform is tailored specifically for Exact GP models combined with any covariance function that employs an exact prediction strategy.

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

Harris Papadopoulos. "Guaranteed Coverage Prediction Intervals with Gaussian Process Regression", in *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 46, no. 12, pp. 9072-9083, Dec. 2024. DOI: [10.1109/TPAMI.2024.3418214](https://doi.org/10.1109/TPAMI.2024.3418214).
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

<a id="1">[1]</a> Harris Papadopoulos. "Guaranteed Coverage Prediction Intervals with Gaussian Process Regression", in *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 46, no. 12, pp. 9072-9083, Dec. 2024. DOI: [10.1109/TPAMI.2024.3418214](https://doi.org/10.1109/TPAMI.2024.3418214). 
([arXiv version](https://arxiv.org/abs/2310.15641))

<a id="2">[2]</a> Vladimir Vovk, Alexander Gammerman, and Glenn Shafer. *Algorithmic Learning in a Random World*, 2nd Ed. Springer, 2023. DOI: [10.1007/978-3-031-06649-8](https://doi.org/10.1007/978-3-031-06649-8).


- - -

Author: Harris Papadopoulos (h.papadopoulos@frederick.ac.cy) / 
Copyright 2024 Harris Papadopoulos / 
License: BSD 3 clause
