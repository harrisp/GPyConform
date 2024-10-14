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


def update_chol_factor(L, b, c, new_factor):
    """
    Update Cholesky factor L with new observation vector b and variance c.
    
    
    Parameters
    ----------
    L : LinearOperator
        Current Cholesky factor of the covariance matrix
    b : torch.Tensor
        Cross-covariance vector between new data point and existing data
    c : float
        Variance of the new data point
    new_factor : torch.Tensor
        Preallocated tensor with the values of L


    Returns
    -------
    newL : LinearOperator
        The updated Cholesky factor
    """

    y = L.solve(b).squeeze_(-1)
    d = torch.sqrt(c - torch.dot(y, y))
    
    n = y.size(0)

    # Update the preallocated tensor
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
            assert test_mean.device == device
            assert test_train_covar.device == device
            assert test_test_covar_diag.device == device
            assert self.train_labels.device == device

            if confs is None:
                confs = torch.tensor([0.95], device=device)
            elif isinstance(confs, np.ndarray):
                confs = torch.tensor(confs, device=device)
            elif isinstance(confs, list):
                confs = torch.tensor(confs, device=device)
            elif not isinstance(confs, torch.Tensor):
                raise ValueError("Confs must be a numpy array, list, or torch tensor")
            else:
                confs = confs.to(device)

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

    PIs = torch.zeros(len(confs), test_mean.size(-1), 2, device=device)
    power = 1 - 1 / gamma
    train_size = L.size(-1)
    identity = torch.eye(train_size+1, dtype=dtype, device=device)
    inf_ninf_tensor = torch.tensor([float('inf'), float('-inf')], dtype=dtype, device=device)
    new_factor = torch.zeros((train_size+1, train_size+1), device=device, dtype=dtype)
    new_factor[:train_size, :train_size].copy_(L.to_dense())

    for i in range(test_mean.size(-1)):
        # Calculate the updated cholesky factor
        b = test_train_covar[..., i, :]
        c = test_test_covar_diag[..., i]
        newL = update_chol_factor(L, b, c, new_factor)

        A = newL._cholesky_solve(Amult, upper=False).squeeze_(-1)
        B = newL._cholesky_solve(Bmult, upper=False).squeeze_(-1)

        D = torch.pow(newL._cholesky_solve(identity, upper=False).diagonal(dim1=-1, dim2=-2), power)
        A /= D
        B /= D
        
        # Element-wise modification of A and B
        modifier = ((2 * (B >= 0)) - 1)
        A *= modifier
        B *= modifier

        # Extract the test example and remove it from A and B
        Atest = A[-1].item()
        Btest = B[-1].item()
        A = A[:-1]
        B = B[:-1]

        listLl = []
        listRr = []
        
        mask_lt = B < Btest
        mask_gt = B > Btest

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
            
        if Btest != 0:
            xx = B == Btest
            Axx = A[xx]
            Bxx = B[xx]
            Pxx = - (Axx + Atest) / (2 * Bxx)

            greater = Axx > Atest
            lesser = Axx < Atest
            equal = Axx == Atest

            listLl.extend([Pxx[greater], -torch.inf + torch.zeros_like(Pxx[lesser]), -torch.inf + torch.zeros_like(Pxx[equal])])
            listRr.extend([torch.inf + torch.zeros_like(Pxx[greater]), Pxx[lesser], torch.inf + torch.zeros_like(Pxx[equal])])
        else:
            condition = (B == 0) & (torch.abs(A) >= torch.abs(Atest))
            listLl.append(-torch.inf + torch.zeros_like(A[condition]))
            listRr.append(torch.inf + torch.zeros_like(A[condition]))

        # Concatenate lists of tensors -> Ll and Rr are tensors
        Ll = torch.cat(listLl)
        Rr = torch.cat(listRr)

        # Add mean of test instance
        Ll += test_mean[i]
        Rr += test_mean[i]
            
        P = torch.unique(torch.cat([Ll, Rr, inf_ninf_tensor]), sorted=True)

        Ll.unsqueeze_(1)
        Rr.unsqueeze_(1)
    
        Llcount = (Ll == P).sum(dim=0)
        Rrcount = (Rr == P).sum(dim=0)

        M = torch.zeros(P.numel(), device=P.device)
        M[0] = 1
        M += Llcount
        M[1:] -= Rrcount[:-1]

        M = M.cumsum(0)
        M /= train_size+1

        for j in range(len(confs)):
            Mbig = M > 1-confs[j]
            indices_of_ones = torch.nonzero(Mbig).squeeze_(-1)
                
            if indices_of_ones.numel() > 0:
                min_index = indices_of_ones.min().item()
                max_index = indices_of_ones.max().item()
            
                PIs[j,i,0] = P[min_index]
                PIs[j,i,1] = P[max_index]
            else:
                max_M = M.max()
                max_indices = torch.nonzero(M == max_M).flatten()
                min_index = max_indices.min().item()
                max_index = max_indices.max().item()
                Point = (P[min_index] + P[max_index]) / 2
                PIs[j,i,0] = Point
                PIs[j,i,1] = Point
        
    return PredictionIntervals(confs, PIs)


def _prediction_regions_asymmetric(self, test_mean: Tensor, test_train_covar: LinearOperator, test_test_covar_diag: Tensor, L: LinearOperator, Amult: Tensor, Bmult: Tensor, confs: Tensor, gamma: float):
    device = L.device
    dtype = L.dtype

    PIs = torch.zeros(len(confs), test_mean.size(-1), 2, device=device)
    power = 1 - 1 / gamma
    train_size = L.size(-1)
    HalfDeltas = (1-confs)/2
    Llindex = (HalfDeltas * (train_size + 1)).floor().to(torch.int64) - 1
    Uuindex = (1 - HalfDeltas * (train_size + 1)).ceil().to(torch.int64) - 1
    Ll = torch.zeros(train_size, dtype=dtype, device=device)
    Uu = torch.zeros(train_size, dtype=dtype, device=device)
    identity = torch.eye(train_size + 1, dtype=dtype, device=device)
    new_factor = torch.zeros((train_size + 1, train_size + 1), device=device, dtype=dtype)
    new_factor[:train_size, :train_size].copy_(L.to_dense())
    neg_inf = float('-inf')
    pos_inf = float('inf')

    for i in range(test_mean.size(-1)):
        # Initialize lower (l_i) and upper (u_i)
        Ll.fill_(neg_inf)
        Uu.fill_(pos_inf)
        
        # Calculate the updated cholesky factor
        b = test_train_covar[..., i, :]
        c = test_test_covar_diag[..., i]
        newL = update_chol_factor(L, b, c, new_factor)

        A = newL._cholesky_solve(Amult, upper=False).squeeze_(-1)
        B = newL._cholesky_solve(Bmult, upper=False).squeeze_(-1)

        D = torch.pow(newL._cholesky_solve(identity, upper=False).diagonal(dim1=-1, dim2=-2), power)
        A /= D
        B /= D
        
        # Extract the test example and remove it from A and B
        Atest = A[-1].item()
        Btest = B[-1].item()
        A = A[:-1]
        B = B[:-1]

        # Condition: B < Btest
        mask_lt = B < Btest
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

def apply_patches():
    # Check if patch has already been applied
    if not hasattr(DefaultPredictionStrategy, 'orig_exact_prediction'):
        # Save original methods as class attributes
        DefaultPredictionStrategy.orig_exact_prediction = DefaultPredictionStrategy.exact_prediction

        # Apply monkey patches
        DefaultPredictionStrategy.exact_prediction = default_exact_prediction
        DefaultPredictionStrategy._prediction_regions_symmetric = _prediction_regions_symmetric
        DefaultPredictionStrategy._prediction_regions_asymmetric = _prediction_regions_asymmetric
