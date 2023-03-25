'''Dirichlet Multinomial Mixture Model'''

# Author: Ziqi Rong <ziqirong@umich.edu>
# License: BSD 3 clause

import numpy as np
import torch
import scipy.special

from sklearn.mixture._base import BaseMixture

'''
TODO
Bad Function
Re-implement this ASAP.
'''
def _initialize_alpha(X, resp):
    n_components = resp.shape[1]
    n_features = X.shape[1]
    max_sum_alpha = 1e3
    alphas = np.zeros([n_components, n_features])
    for k in range(n_components):
        data_k = X[np.where(resp[:,k]==1)[0], :].T
        if (data_k.sum() == 0):
            alphas[k] = np.zeros(n_features)
            continue
        p_cluster = data_k.sum(axis=1)/data_k.sum()#p
        zero_idx = np.argwhere(np.all(data_k[..., :] == 0, axis=0))
        data_k = np.delete(data_k, zero_idx, axis=1)
        p_cell = data_k/data_k.sum(axis=0)#ppp
        p_E = p_cell.mean(axis=1)#pp
        if (p_cell.shape[1] == 1):
            p_var = p_cell.var(axis=1, ddof=0)#In case that only one cell in a cluster.
        else:
            p_var = p_cell.var(axis=1, ddof=1)
        p_var[np.where(p_var==0)] = p_var.mean()
        sum_alpha = 0
        for i in range(n_features - 1):#Why G-1???
            # print(p_E[i]*(1-p_E[i])/p_var[i] - 1, p_E[i], p_var[i], i)
            if ((p_E[i] != 0) and (p_var[i] != 0)):#In case that varience are all zero.
                tmp = p_E[i]*(1-p_E[i])/p_var[i] - 1
                tmp = max(tmp, 1e-4)
                sum_alpha = sum_alpha + (1/(n_features-1))*np.log(tmp)
        sum_alpha = np.exp(sum_alpha)
        if (sum_alpha == 1):#In case that varience are all zero.
            sum_alpha = max_sum_alpha
        sum_alpha = min(sum_alpha, max_sum_alpha)#In case that alpha is too big, so that loglik calculation will overflow.
        alphas[k] = sum_alpha * p_cluster
        alphas[k] = alphas[k] + 1e-4
    return alphas

def _estimate_log_dirichlet_multinomial_prob(X, alphas, pytorch):
    n_samples, n_features = X.shape
    n_components, _ = alphas.shape
    if pytorch == 0:
        term1 = scipy.special.gammaln(X + alphas[:,np.newaxis]).sum(axis=2) - scipy.special.gammaln(alphas[:,np.newaxis]).sum(axis=2)
        term2 = - scipy.special.gammaln(X.sum(axis=1)+alphas[:,np.newaxis].sum(axis=2)) + scipy.special.gammaln(alphas.sum(axis=1))[:,np.newaxis]
        log_prob = (term1+term2).T
    else:
        if pytorch == 1:
            if isinstance(X, np.ndarray):
                X = torch.from_numpy(X)
            if isinstance(alphas, np.ndarray):
                alphas = torch.from_numpy(alphas)
        elif pytorch == 2:
            if isinstance(X, np.ndarray):
                X = torch.from_numpy(X).to('cuda')
            if isinstance(alphas, np.ndarray):
                alphas = torch.from_numpy(alphas).to('cuda')
        term1 = torch.special.gammaln(X + alphas[:,None]).sum(axis=2) - torch.special.gammaln(alphas[:,None]).sum(axis=2)
        term2 = - torch.special.gammaln(X.sum(axis=1)+alphas[:,None].sum(axis=2)) + torch.special.gammaln(alphas.sum(axis=1))[:,None]
        log_prob = (term1+term2).T
        log_prob = log_prob.cpu().numpy()
    return log_prob

def _estimate_dirichlet_multinomial_alphas(X, resp, alphas, reg_alpha, pytorch):
    if pytorch == 0:
        T = X.sum(axis=1)
        term1 = ((X / (X - 1 + alphas[:,np.newaxis])) * np.transpose(resp.T[:,np.newaxis], axes=(0,2,1))).sum(axis=1)
        term2 = ((T / (T - 1 + alphas.sum(axis=1)[:,np.newaxis])) * resp.T).sum(axis=1)
        alphas = alphas * (term1/term2[:,np.newaxis]) + reg_alpha
    else:
        if pytorch == 1:
            if isinstance(X, np.ndarray):
                X = torch.from_numpy(X)
            if isinstance(resp, np.ndarray):
                resp = torch.from_numpy(resp)
            if isinstance(alphas, np.ndarray):
                alphas = torch.from_numpy(alphas)
        elif pytorch == 2:
            if isinstance(X, np.ndarray):
                X = torch.from_numpy(X).to('cuda')
            if isinstance(resp, np.ndarray):
                resp = torch.from_numpy(resp).to('cuda')
            if isinstance(alphas, np.ndarray):
                alphas = torch.from_numpy(alphas).to('cuda')
        T = X.sum(axis=1)
        term1 = ((X / (X - 1 + alphas[:,None])) * torch.transpose(resp.T[:,None], 1, 2)).sum(axis=1)
        term2 = ((T / (T - 1 + alphas.sum(axis=1)[:,None])) * resp.T).sum(axis=1)
        alphas = alphas * (term1/term2[:,None]) + reg_alpha
        alphas = alphas.cpu().numpy()
    return alphas

class DirichletMultinomialMixture(BaseMixture):
    """Dirichlet-multinomial Mixture.

    Parameters
    ----------
    n_components : int, default=1.
        The number of mixture components.
    
    tol : float, default=1e-3.
        The convergence threshold. EM iterations will stop when the
        lower bound average gain is below this threshold.
    
    reg_covar : float, default=1e-6.
    
    max_iter : int, default=10.
        The number of EM iterations to perform.
    
    n_init : int, default=1.
    
    init_params : str, default='kmeans'.
        The method used to initialize the weights, the means and the
        precisions.
        String must be one of:

        - 'kmeans' : responsibilities are initialized using kmeans.
        - 'k-means++' : use the k-means++ method to initialize.
        - 'random' : responsibilities are initialized randomly.
        - 'random_from_data' : initial means are randomly selected data points.
    
    random_state : int, default=None.
    
    warm_start : bool, default=False.
    
    verbose : int, default=0.
        Enable verbose output. If 1 then it prints the current
        initialization and each iteration step. If greater than 1 then
        it prints also the log probability and the time needed
        for each step.
    
    verbose_interval : int, default=10.
        Number of iteration done before the next print.
    
    reg_alpha : float, default=1e-10.
    
    pytorch : int, default=0.
        Whether to use PyTorch in EM iterations.
        Must be one of:

        - 0 : Do not use PyTorch (Use NumPy as default).
        - 1 : Use PyTorch tensor on cpu.
        - 2 : Use Pytorch tensor on cuda.

    Attributes
    ----------
    weights : array-like of shape (n_components,)
        The weights of each mixture components.

    alphas : array-like of shape (n_components, n_features)
        The alphas of each mixture component.

    """
    def __init__(
        self,
        n_components=1,
        tol=1e-3,
        reg_covar=1e-6,
        max_iter=10,
        n_init=1,
        init_params="kmeans",
        random_state=None,
        warm_start=False,
        verbose=0,
        verbose_interval=10,
        reg_alpha = 1e-10,
        pytorch = 0
    ):
        if pytorch not in [0,1,2]:
            raise ValueError('Invalid argument \'pytorch\'. It could only be 0, 1 or 2.')
        if (pytorch == 2) and (not torch.cuda.is_available()):
            raise RuntimeError('Cuda not available.')
        super().__init__(
            n_components=n_components,
            tol=tol,
            reg_covar=reg_covar,
            max_iter=max_iter,
            n_init=n_init,
            init_params=init_params,
            random_state=random_state,
            warm_start=warm_start,
            verbose=verbose,
            verbose_interval=verbose_interval,
        )
        self.reg_alpha = reg_alpha
        self.pytorch = pytorch
    
    def _check_parameters(self, X):
        pass

    def _initialize(self, X, resp):
        nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
        self.alphas = _initialize_alpha(X, resp)
        self.weights = nk/nk.sum()

    def _estimate_log_prob(self, X):
        return _estimate_log_dirichlet_multinomial_prob(
            X, self.alphas, self.pytorch
        )
    
    def _estimate_log_weights(self):
        return np.log(self.weights)

    def _m_step(self, X, log_resp):
        n_samples, n_features = X.shape
        n_components, _ = log_resp.shape
        resp = np.exp(log_resp)
        self.weights = resp.sum(axis=0)/resp.sum()
        self.alphas = _estimate_dirichlet_multinomial_alphas(X, resp, self.alphas, self.reg_alpha, self.pytorch)

    def _compute_lower_bound(self, _, log_prob_norm):
        return log_prob_norm

    def _get_parameters(self):
        return (
            self.weights,
            self.alphas,
        )

    def _set_parameters(self, params):
        (
            self.weights,
            self.alphas,
        ) = params