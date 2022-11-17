'''Dirichlet Multinomial Mixture Model'''

# Author: Ziqi Rong <ziqirong01@gmail.com>
# License: 

import numpy as np

from scipy.special import gammaln

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

def _estimate_log_dirichlet_multinomial_prob(X, alphas):
    n_samples, n_features = X.shape
    n_components, _ = alphas.shape
    term1 = gammaln(X + alphas[:,np.newaxis]).sum(axis=2) - gammaln(alphas[:,np.newaxis]).sum(axis=2)
    term2 = - gammaln(X.sum(axis=1)+alphas[:,np.newaxis].sum(axis=2)) + gammaln(alphas.sum(axis=1))[:,np.newaxis]
    log_prob = (term1+term2).T
    return log_prob

def _estimate_dirichlet_multinomial_alphas(X, resp, alphas, reg_alpha):
    T = X.sum(axis=1)
    term1 = ((X / (X - 1 + alphas[:,np.newaxis])) * np.transpose(resp.T[:,np.newaxis], axes=(0,2,1))).sum(axis=1)
    term2 = ((T / (T - 1 + alphas.sum(axis=1)[:,np.newaxis])) * resp.T).sum(axis=1)
    alphas = alphas * (term1/term2[:,np.newaxis]) + reg_alpha
    return alphas

class DirichletMultinomialMixture(BaseMixture):
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
        reg_alpha = 1e-10
    ):
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
    
    def _check_parameters(self, X):
        pass

    def _initialize(self, X, resp):
        nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
        self.alphas = _initialize_alpha(X, resp)
        self.weights = nk/nk.sum()

    def _estimate_log_prob(self, X):
        return _estimate_log_dirichlet_multinomial_prob(
            X, self.alphas
        )
    
    def _estimate_log_weights(self):
        return np.log(self.weights)

    def _m_step(self, X, log_resp):
        n_samples, n_features = X.shape
        n_components, _ = log_resp.shape
        resp = np.exp(log_resp)
        self.weights = resp.sum(axis=0)/resp.sum()
        self.alphas = _estimate_dirichlet_multinomial_alphas(X, resp, self.alphas, self.reg_alpha)

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