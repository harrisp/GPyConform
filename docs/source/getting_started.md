# Getting Started

## Installation

From [PyPI](https://pypi.org/project/gpyconform/)

```bash
pip install gpyconform
```

From [conda-forge](https://anaconda.org/conda-forge/gpyconform)

```bash
conda install conda-forge::gpyconform
```

## Tutorial

Let us illustrate how we may use GPyConform for obtaining Conformal Prediction Intervals on a simple function. In particular we follow the 
Simple GP Regression tutorial of GPyTorch 
(see: [GPyTorch Regression Tutorial](https://docs.gpytorch.ai/en/stable/examples/01_Exact_GPs/Simple_GP_Regression.html)) in order to 
demonstrate the functionality of GPyConform.

We'll be training an RBF kernel Gaussian Process on the function

\begin{align}
  y &= \sin(2\pi x) + \epsilon \\
  \epsilon &\sim \mathcal{N}(0, 0.04) 
\end{align}

with 500 training examples, and testing on 5000 test examples.

### Imports and data

First, let's import the needed packages and set up the data. For reproducibility we set the random seed to 42. The training set is generated as 
500 random points uniformly distributed in [0,1], on which we evaluate the function and add Gaussian noise to get the training labels. 
The test set is generated in the same way, with 5000 random points:

```python
import math
import torch
import gpytorch
import gpyconform

torch.manual_seed(42)

train_x = torch.rand(500)
train_y = torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * math.sqrt(0.04)

test_x = torch.rand(5000)
test_y = torch.sin(test_x * (2 * math.pi)) + torch.randn(test_x.size()) * math.sqrt(0.04)
```

### Setting up the model

The model is constructed in exactly the same way as with GPyTorch, but using `gpyconform.ExactGPCP` instead of `gpytorch.models.ExactGP`. 
For details please refer to the [GPyTorch documentation](https://gpytorch.readthedocs.io/en/latest/).

```python
class ExactGPCPModel(gpyconform.ExactGPCP):
    def __init__(self, train_x, train_y, likelihood, cpmode='symmetric'):
        super(ExactGPCPModel, self).__init__(train_x, train_y, likelihood, cpmode=cpmode)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPCPModel(train_x, train_y, likelihood)
```

**Note:** Any mean function from `gpytorch.means` and any kernel function that employs an exact prediction 
strategy from `gpytorch.kernels` can be used with GPyConform.

### Model modes

Like the GPyTorch `ExactGP` module, `ExactGPCP` also has a `.train()` and `.eval()` mode:
- `.train()` mode is for optimizing model hyperameters.
- `.eval()` mode is for computing the Conformal Prediction Intervals and the original GP predictions through the model posterior.

### Training the model

The hyperparameter training of the Gaussian Process is also performed in exactly the same way as GPyTorch.

```python
training_iter = 50

model.train()
likelihood.train()

optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

for i in range(training_iter):
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
        i + 1, training_iter, loss.item(),
        model.covar_module.base_kernel.lengthscale.item(),
        model.likelihood.noise.item()
    ))
    optimizer.step()
```

```text
Iter 1/50 - Loss: 0.882   lengthscale: 0.693   noise: 0.693
Iter 2/50 - Loss: 0.841   lengthscale: 0.644   noise: 0.644
Iter 3/50 - Loss: 0.795   lengthscale: 0.598   noise: 0.598
...
Iter 48/50 - Loss: -0.167   lengthscale: 0.318   noise: 0.030
Iter 49/50 - Loss: -0.170   lengthscale: 0.323   noise: 0.030
Iter 50/50 - Loss: -0.172   lengthscale: 0.328   noise: 0.031
```

### Making predictions: Obtaining prediction intervals

Like in GPyTorch, to make predictions with the model we put it in `.eval()` mode.

GPyConform has an additional parameter named `cpmode`, which determines the Conformal Prediction approach:
- `.cpmode='symmetric'` (default) employs the absolute residual nonconformity measure approach as described in [1].
- `.cpmode='asymmetric'` employs the asymmetric version of the nonconformity measure defined in [1], following the approach described in Chapter 2.3 of [2].
- `.cpmode=None` reverts to GPyTorch's ExactGP behavior (for details refer to the GPyTorch documentation).

**Note:** The ``cpmode`` property can change at any time without affecting the model. For `.cpmode=None` we should also put the likelihood in `.eval()` mode 
and call both modules on the test data.

After putting the model in `.eval()` mode, and if needed changing `cpmode`, we can then call the model on the test data to obtain the 
Conformal Prediction Intervals. In addition to the test data, the model (in `.eval()` mode) has the following optional parameters:
- `gamma`: the gamma parameter of the nonconformity measure (float). The default value is 2.
- `confs`: a list of the confidence levels for which to return Prediction Intervals (torch.Tensor, numpy.array, or list). The default is [0.95].

A trained GPyConform model in `.eval()` mode (with `cpmode` set to either `symmetric` or `asymmetric`) returns a `PredictionIntervals` object containing the 
Prediction Intervals for all confidence levels in `confs`. The `PredictionIntervals` object allows direct access to the Prediction Intervals as well as their 
evaluation in terms of empirical calibration and informational efficiency.

So let's put the model in `.eval()` mode and obtain the `PredictionIntervals` object with `gamma` set to 2 for the 99%, 95% and 90% confidence levels.

```python
model.eval()

with torch.no_grad():  # Disable gradient calculation
    pis = model(test_x, gamma=2, confs=[0.99, 0.95, 0.9])
```

We can then extract a torch tensor with the prediction intervals for any of the three confidence levels (here 95%):

```python
print(pis(0.95))
```

```text
tensor([[ 0.4023,  1.1818],
        [-1.2279, -0.4469],
        [-1.3555, -0.5769],
        ...,
        [-1.3729, -0.5945],
        [ 0.2350,  1.0155],
        [ 0.3880,  1.1654]])
```
The output is a tensor with a row for each test instance, and where the two columns specify the lower and upper bound of each Prediction Interval.

We can also extract a dictionary with confidence levels as keys and the corresponding Prediction Interval tensors as values:

```python
print(pis())
```

```text
{
   0.9: tensor([[ 0.4642,  1.1230],
        	[-1.1645, -0.5086],
        	[-1.2947, -0.6365],
        	...,
        	[-1.3116, -0.6540],
        	[ 0.2972,  0.9568],
        	[ 0.4483,  1.1060]]),
  0.95: tensor([[ 0.4023,  1.1818],
        	[-1.2279, -0.4469],
        	[-1.3555, -0.5769],
        	...,
        	[-1.3729, -0.5945],
        	[ 0.2350,  1.0155],
        	[ 0.3880,  1.1654]]),
  0.99: tensor([[ 0.2844,  1.2996],
        	[-1.3431, -0.3283],
        	[-1.4742, -0.4584],
        	...,
        	[-1.4912, -0.4757],
        	[ 0.1181,  1.1335],
        	[ 0.2733,  1.2876]])
}
```

Aditionally, we can evaluate the Prediction Intervals (here using all available metrics, which is the default), at any of the three confidence levels (here 99%):

```python
print(pis.evaluate(0.99, y=test_y))
```

```text
{'mean_width': 1.01625, 'median_width': 1.01515, 'error': 0.01020}
```
The output is a dictionary with the metrics as keys.

To obtain the corresponding `'asymmetric'` prediction intervals, we only set `.cpmode='asymmetric'`:

```python
model.cpmode = 'asymmetric'

with torch.no_grad():  # Disable gradient calculation
    pis = model(test_x, gamma=2, confs=[0.99, 0.95, 0.9])
```

We can extract and/or evaluate the Prediction Intervals in the same way as above. E.g. to obtain the mean and median widths of the 99% Prediction Intervals:

```python
pis.evaluate(0.99, ['mean_width', 'median_width'])
```

```text
{'mean_width': 1.08704, 'median_width': 1.08623}
```
Note that in this case, since the 'error' metric is not required, we do not need to provide the true target values.