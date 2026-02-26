#!/usr/bin/env python3

import torch
import linear_operator
import numpy as np

from linear_operator.operators import LinearOperator
from torch import Tensor
from gpytorch import settings
from gpytorch.models.exact_prediction_strategies import (
     DefaultPredictionStrategy
 )
from gpyconform.prediction_intervals import PredictionIntervals
from gpyconform.constants import _CONF_SCALE, _GAMMA_INF_THRESHOLD

def update_chol_factor(L, b, c, new_factor):
    """
    Update Cholesky factor L with new observation vector b and variance c.
    
    
    Parameters
    ----------
    L : LinearOperator
        Current Cholesky factor of the covariance matrix
    b : torch.Tensor
        Cross-covariance vector between new data point and existing data
    c : float or torch.Tensor of shape ()
        Variance of the new data point
    new_factor : torch.Tensor
        Preallocated tensor with the values of L


    Returns
    -------
    newL : LinearOperator
        The updated Cholesky factor
    """

    y = L.solve(b)  # keep 1D or 2D rhs; L.solve handles both
    if y.dim() > 1 and y.size(-1) == 1:
        y = y.squeeze(-1)
    # Stabilize Schur complement
    s = c - torch.dot(y, y)
    eps = torch.finfo(s.dtype).eps * 32
    d = torch.sqrt((s + eps).clamp_min(0))  # jitter + clamp

    n = y.size(0)
    new_factor[n, :n].copy_(y)
    new_factor[n, n] = d
    new_factor[:n, n].zero_()
    
    return linear_operator.to_linear_operator(new_factor)


def default_exact_prediction(self, joint_mean, joint_covar, **kwargs):
    if not kwargs:
        return DefaultPredictionStrategy.orig_exact_prediction(self, joint_mean, joint_covar)
    else:
        cpmode = kwargs.pop('cpmode', 'symmetric')
        gamma = kwargs.pop('gamma', 2)
        confs = kwargs.pop('confs', None)
        
        with torch.no_grad():
            # Find the components of the distribution that contain test data
            test_mean = joint_mean[..., self.num_train :]
            # For efficiency - we can make things more efficient
            if joint_covar.size(-1) <= settings.max_eager_kernel_size.value():
                test_covar = joint_covar[..., self.num_train :, :].to_dense()
                test_test_covar_diag = torch.diagonal(test_covar[..., self.num_train :], dim1=-2, dim2=-1)
                test_train_covar = test_covar[..., : self.num_train]
            else:
                test_test_covar_diag = torch.diagonal(joint_covar[..., self.num_train :, self.num_train :], dim1=-2, dim2=-1)
                test_train_covar = joint_covar[..., self.num_train :, : self.num_train]

            if test_train_covar.ndimension() > 2:
                raise ValueError("Batches of inputs are currently not supported.")

            # Calculate the training mean and Cholesky decomposition of the covariance matrix (KsI):
            mvn = self.likelihood(self.train_prior_dist, self.train_inputs)
            train_mean, L = mvn.loc, mvn.lazy_covariance_matrix.cholesky(upper=False)

            # Ensure all input tensors are on the same device as L
            device = L.device
            dtype = L.dtype
            test_mean = test_mean.to(device=device, dtype=dtype)
            test_test_covar_diag = test_test_covar_diag.to(device=device, dtype=dtype)
            self.train_labels = self.train_labels.to(device=device, dtype=dtype)
            if not isinstance(test_train_covar, LinearOperator):
                test_train_covar = test_train_covar.to(device=device, dtype=dtype)

            if confs is None:
                confs = torch.tensor([0.95], device=device, dtype=torch.float64)
            elif isinstance(confs, (np.ndarray, list, tuple)):
                confs = torch.tensor(confs, device=device, dtype=torch.float64)
            elif isinstance(confs, torch.Tensor):
                confs = confs.to(device=device, dtype=torch.float64)
            else:
                raise TypeError("'confs' must be a torch.Tensor, numpy.ndarray, list, or tuple.")

            if confs.numel() == 0:
                raise ValueError("'confs' must contain at least one confidence level.")

            if not torch.all((confs > 0.0) & (confs < 1.0)):
                raise ValueError("All confidence levels in 'confs' must be strictly in (0, 1).")

            # Add noise to the test covariance diagonal
            test_test_covar_diag += self.likelihood.noise
        
            # Calculate y_i as difference from the mean
            train_labels_offset = self.train_labels - train_mean

            # Amult = (y_1,...,y_n,0), where y_i is the difference from train_mean, and Bmult = (0,...,0,1)
            Amult = torch.cat([train_labels_offset, torch.tensor([0], device=train_labels_offset.device, dtype=train_labels_offset.dtype)], dim=-1).unsqueeze_(-1)
            Bmult = torch.zeros_like(Amult)
            Bmult[-1] = 1

            if cpmode == 'symmetric':
                return self._prediction_regions_symmetric(test_mean, test_train_covar, test_test_covar_diag, L, Amult, Bmult, confs, gamma)
            elif cpmode == 'asymmetric':
                return self._prediction_regions_asymmetric(test_mean, test_train_covar, test_test_covar_diag, L, Amult, Bmult, confs, gamma)
            else:
                raise ValueError(f"The setting {cpmode} for cpmode is not valid. Possible settings are 'symmetric' or 'asymmetric'.")


def _prediction_regions_symmetric(self, test_mean: Tensor, test_train_covar: LinearOperator, test_test_covar_diag: Tensor, L: LinearOperator, Amult: Tensor, Bmult: Tensor, confs: Tensor, gamma: float):
    device = L.device
    dtype = L.dtype
    eps = torch.finfo(torch.float64).eps * 32
    
    PIs = torch.zeros(len(confs), test_mean.size(-1), 2, device=device, dtype=torch.float64)
    if gamma >= _GAMMA_INF_THRESHOLD:
        power = None
    else:
        gamma = float(gamma)
        power = torch.as_tensor((gamma - 1.0)/gamma, device=L.device, dtype=torch.float64)
        
    train_size = L.size(-1)
    train_size1 = train_size + 1
    confs_int = torch.round(confs * _CONF_SCALE).to(torch.int64)
    identity = torch.eye(train_size1, device=device, dtype=dtype)
    new_factor = torch.zeros((train_size1, train_size1), device=device, dtype=dtype)
    new_factor[:train_size, :train_size].copy_(L.to_dense())
    inf_ninf_tensor = torch.tensor([float('inf'), float('-inf')], device=device, dtype=torch.float64)
    zero64 = torch.zeros((), device=device, dtype=torch.float64)
    test_mean = test_mean.double()
    
    for i in range(test_mean.size(-1)):
        # Calculate the updated cholesky factor
        b = test_train_covar[..., i, :]
        c = test_test_covar_diag[..., i]
        newL = update_chol_factor(L, b, c, new_factor)

        A = newL._cholesky_solve(Amult, upper=False).squeeze_(-1).double()
        B = newL._cholesky_solve(Bmult, upper=False).squeeze_(-1).double()

        D = newL._cholesky_solve(identity, upper=False).diagonal(dim1=-1, dim2=-2).double()
        if power is not None:
            D = torch.pow(D, power)

        A /= D
        B /= D
        
        # Element-wise modification of A and B
        modifier = torch.where(B >= 0, torch.ones_like(B), -torch.ones_like(B))
        A *= modifier
        B *= modifier

        # Extract the test example and remove it from A and B
        Atest = A[-1]
        Btest = B[-1]
        A = A[:-1]
        B = B[:-1]

        listLl = []
        listRr = []
        
        mask_eq = torch.isclose(B, Btest, atol=eps, rtol=0.0)    
        mask_lt = (B < Btest) & ~mask_eq
        mask_gt = (B > Btest) & ~mask_eq

        if mask_lt.any():
            Axx = A[mask_lt]
            Bxx = B[mask_lt]
            P1xx = - (Axx - Atest) / (Bxx - Btest)
            P2xx = - (Axx + Atest) / (Bxx + Btest)
            listLl.append(torch.min(P1xx, P2xx))
            listRr.append(torch.max(P1xx, P2xx))
            
        if mask_gt.any():
            Axx = A[mask_gt]
            Bxx = B[mask_gt]
            P1xx = - (Axx - Atest) / (Bxx - Btest)
            P2xx = - (Axx + Atest) / (Bxx + Btest)
            min_Pxx = torch.min(P1xx, P2xx)
            max_Pxx = torch.max(P1xx, P2xx)
            listLl.extend([-torch.inf + torch.zeros_like(min_Pxx), max_Pxx])
            listRr.extend([min_Pxx, torch.inf + torch.zeros_like(max_Pxx)])
            
        if not torch.isclose(Btest, zero64, atol=eps, rtol=0.0):
            Axx = A[mask_eq]
            Bxx = B[mask_eq]
            Pxx = - (Axx + Atest) / (2 * Bxx)

            equal = torch.isclose(Axx, Atest, atol=eps, rtol=0.0)   
            greater = (Axx > Atest)  & ~equal
            lesser = (Axx < Atest) & ~equal

            listLl.extend([Pxx[greater], -torch.inf + torch.zeros_like(Pxx[lesser]), -torch.inf + torch.zeros_like(Pxx[equal])])
            listRr.extend([torch.inf + torch.zeros_like(Pxx[greater]), Pxx[lesser], torch.inf + torch.zeros_like(Pxx[equal])])
        else:
            condition = torch.isclose(B, zero64, atol=eps, rtol=0.0) & (
                (A.abs() > Atest.abs()) |
                torch.isclose(A.abs(), Atest.abs(), atol=eps, rtol=0.0)
                )
            listLl.append(-torch.inf + torch.zeros_like(A[condition]))
            listRr.append(torch.inf + torch.zeros_like(A[condition]))

        # Concatenate lists of tensors -> Ll and Rr are tensors
        Ll = torch.cat(listLl)
        Rr = torch.cat(listRr)

        # Add mean of test instance
        Ll += test_mean[i]
        Rr += test_mean[i]
            
        P = torch.unique(torch.cat([Ll, Rr, inf_ninf_tensor]), sorted=True)

        Ll = Ll.unsqueeze_(1)
        Rr = Rr.unsqueeze_(1)
    
        Llcount = (Ll == P).sum(dim=0)
        Rrcount = (Rr == P).sum(dim=0)

        M = torch.zeros(P.numel(), device=device, dtype=torch.int64)
        M[0] = 1
        M += Llcount
        M[1:] -= Rrcount[:-1]

        M = M.cumsum(0)

        for j in range(len(confs_int)):
            Mbig = M * _CONF_SCALE > (_CONF_SCALE - confs_int[j]) * train_size1
            indices_of_ones = torch.nonzero(Mbig, as_tuple=True)[0]
                
            if indices_of_ones.numel() > 0:
                min_index = indices_of_ones.min()
                max_index = indices_of_ones.max()
            
                PIs[j,i,0] = P[min_index]
                PIs[j,i,1] = P[max_index]
            else:
                max_M = M.max()
                max_indices = torch.nonzero(M == max_M, as_tuple=True)[0]
                min_index = max_indices.min()
                max_index = max_indices.max()
                Point = (P[min_index] + P[max_index]) / 2
                PIs[j,i,0] = Point
                PIs[j,i,1] = Point
    
    return PredictionIntervals(confs, PIs)


def _prediction_regions_asymmetric(self, test_mean: Tensor, test_train_covar: LinearOperator, test_test_covar_diag: Tensor, L: LinearOperator, Amult: Tensor, Bmult: Tensor, confs: Tensor, gamma: float):
    device = L.device
    dtype = L.dtype
    eps = torch.finfo(torch.float64).eps * 32

    PIs = torch.zeros(len(confs), test_mean.size(-1), 2, device=device, dtype=torch.float64)
    if gamma >= _GAMMA_INF_THRESHOLD:
        power = None
    else:
        gamma = float(gamma)
        power = torch.as_tensor((gamma - 1.0)/gamma, device=L.device, dtype=torch.float64)
        
    train_size = L.size(-1)
    train_size1 = train_size + 1
    confs_int = torch.round(confs * _CONF_SCALE).to(torch.int64)
    Llindex = ((_CONF_SCALE - confs_int) * train_size1) // (2 * _CONF_SCALE) - 1
    Uuindex = (((_CONF_SCALE + confs_int) * train_size1 + 2 * _CONF_SCALE - 1) // (2 * _CONF_SCALE)) - 1
    Ll = torch.zeros(train_size, device=device, dtype=torch.float64)
    Uu = torch.zeros(train_size, device=device, dtype=torch.float64)
    identity = torch.eye(train_size1, device=device, dtype=dtype)
    new_factor = torch.zeros((train_size1, train_size1), device=device, dtype=dtype)
    new_factor[:train_size, :train_size].copy_(L.to_dense())
    neg_inf = float('-inf')
    pos_inf = float('inf')
    test_mean = test_mean.double()

    for i in range(test_mean.size(-1)):
        # Initialize lower (l_i) and upper (u_i)
        Ll.fill_(neg_inf)
        Uu.fill_(pos_inf)
        
        # Calculate the updated cholesky factor
        b = test_train_covar[..., i, :]
        c = test_test_covar_diag[..., i]
        newL = update_chol_factor(L, b, c, new_factor)

        A = newL._cholesky_solve(Amult, upper=False).squeeze_(-1).double()
        B = newL._cholesky_solve(Bmult, upper=False).squeeze_(-1).double()

        D = newL._cholesky_solve(identity, upper=False).diagonal(dim1=-1, dim2=-2).double()
        if power is not None:
            D = torch.pow(D, power)

        A /= D
        B /= D
        
        # Extract the test example and remove it from A and B
        Atest = A[-1]
        Btest = B[-1]
        A = A[:-1]
        B = B[:-1]

        # Condition: B < Btest
        mask_eq = torch.isclose(B, Btest, atol=eps, rtol=0.0)
        mask_lt = (B < Btest) & ~mask_eq
        Ll[mask_lt] = (A[mask_lt] - Atest) / (Btest - B[mask_lt])
        Uu[mask_lt] = Ll[mask_lt]

        # Sort Ll and Uu
        Ll, _ = Ll.sort()
        Uu, _ = Uu.sort()

        # Add mean of test instance
        Ll += test_mean[i]
        Uu += test_mean[i]

        for j in range(len(confs)):
            if Llindex[j] < 0:
                PIs[j,i,0] = neg_inf
                PIs[j,i,1] = pos_inf
            else:
                PIs[j,i,0] = Ll[Llindex[j]]
                PIs[j,i,1] = Uu[Uuindex[j]]

    return PredictionIntervals(confs, PIs)


def original_exact_prediction(*args, **kwargs):
    pass

def is_patched():
    return hasattr(DefaultPredictionStrategy, "orig_exact_prediction")

def apply_patches():
    # Check if patch has already been applied
    if is_patched():
        return
    
    # Save original methods as class attributes
    DefaultPredictionStrategy.orig_exact_prediction = DefaultPredictionStrategy.exact_prediction

    # Apply monkey patches
    DefaultPredictionStrategy.exact_prediction = default_exact_prediction
    DefaultPredictionStrategy._prediction_regions_symmetric = _prediction_regions_symmetric
    DefaultPredictionStrategy._prediction_regions_asymmetric = _prediction_regions_asymmetric
