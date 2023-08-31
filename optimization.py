from scipy import stats
import numpy as np
from matrix_operations import vec_matrix, replace_diagonal
from statsmodels.tools.numdiff import approx_fprime_cs, approx_hess_cs

# numerical optimization


def marginal_density(x, parameters):
    regimes, k, residuals, smoothed_prob = parameters
    b_mat = vec_matrix(np.array(x[0:k ** 2]))
    lam_m = np.zeros([regimes-1, k, k])

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
    print(sigma)
    for regime in range(regimes):

        likelihood_array[regime] = (np.exp(smoothed_prob[regime, :]) * stats.multivariate_normal(
                                        mean=None, cov=sigma[regime, :, :]).logpdf(residuals.T).T).sum()
    return -likelihood_array.sum()




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
    return gradient





''' 

 def hessian(params):
        """
        Hessian matrix of the likelihood function, evaluated at the given
        parameters

        Parameters
        ----------
        params : array_like
            Array of parameters at which to evaluate the Hessian
            function.
        transformed : boolean, optional
            Whether or not `params` is already transformed. Default is True.
        """
        params = np.array(params, ndmin=1)

        return approx_hess_cs(params, marginal_density)


def fprime(params, transformed=True):
    """
    Compute the score function at params.

    Parameters
    ----------
    params : array_like
        Array of parameters at which to evaluate the score
        function.
    transformed : boolean, optional
        Whether or not `params` is already transformed. Default is True.
    """
    params = np.array(params, ndmin=1)

    return approx_fprime_cs(params, marginal_density,)

 '''