from scipy import stats #
import numpy as np
from matrix_operations import vec_matrix, replace_diagonal
from sklearn.covariance import LedoitWolf
from statsmodels.tools.numdiff import approx_fprime_cs, approx_hess_cs


# numerical optimization


def marginal_density(x, parameters):
    regimes, k, residuals, smoothed_prob = parameters
    b_mat = vec_matrix(np.array(x[0:k ** 2]))
    lam_m = np.zeros([regimes - 1, k, k])
    mean_zero = np.zeros([k])

    start = k * k
    for m in range(regimes - 1):
        end = start + k
        lam_m[m, :, :] = replace_diagonal(x[start:end])
        start = end

    # create an array of lambdas here ...
    # also include restricted models  ....
    likelihood_array = np.zeros([regimes])
    sigma = np.zeros([regimes, k, k])
    for regime in range(regimes):
        if regime == 0:
            sigma[regime, :, :] = b_mat @ b_mat.T
        else:
            sigma[regime, :, :] = b_mat @ lam_m[regime - 1, :, :] @ b_mat.T
        x = np.random.multivariate_normal(mean=mean_zero, cov=sigma[regime, :, :], size=50, check_valid='ignore')
        cov = LedoitWolf().fit(x)
        sigma[regime, :, :] = cov.covariance_

    for regime in range(regimes):
        # Note that I am taking the exp of smoothed probability because original smoothed prob. are in log this
        likelihood_array[regime] = (np.exp(smoothed_prob[regime, :]) * stats.multivariate_normal(
            mean=None, cov=sigma[regime, :, :], allow_singular=True).logpdf(residuals.T).T).sum()
    return -likelihood_array.sum()

''' 
def gradient_respecting_bounds(bounds, fun, eps=1e-8):
    """bounds: list of tuples (lower, upper)"""

    def gradient(x):
        fx = fun(x)
        grad = np.zeros(len(x))
        for k in range(len(x)):
            d = np.zeros(len(x))
            d[k] = eps if x[k] + eps <= bounds[k][1] else -eps
            grad[k] = (fun(x + d) - fx) / d[k]
        return grad

        return gradient(x)
'''

''' 
 def hessian(params):
     params = np.array(params, ndmin=1)
     return approx_hess_cs(params, marginal_density)


def fprime(params, transformed=True):
    params = np.array(params, ndmin=1)
    return approx_fprime_cs(params, marginal_density)

'''


def b_matrix_sigma(x, k, regimes):
    b_mat = vec_matrix(np.array(x[0:k ** 2]))
    lam_m = np.zeros([regimes - 1, k, k])
    start = k * k
    for m in range(regimes - 1):
        end = start + k
        lam_m[m, :, :] = replace_diagonal(x[start:end])
        start = end

    sigma = np.zeros([regimes, k, k])
    for regime in range(regimes):
        if regime == 0:
            sigma[regime, :, :] = b_mat @ b_mat.T
        else:
            sigma[regime, :, :] = b_mat @ lam_m[regime - 1, :, :] @ b_mat.T

    return b_mat, sigma
