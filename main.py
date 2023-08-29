import numpy as np
from scipy.special import logsumexp
from matrix_operations import vec_matrix
from scipy import stats
import data as data

'''
import pandas as pd
from scipy import stats
from sklearn.mixture import GaussianMixture
'''

# Model
regimes = 2
lags = 2
beta = np.array([0, 0, 0, 1]).reshape((-1, 1))  # must be shaped according to (number of variables, 1)
data.data_matrix(data.df, lags, beta)


def initialize(number_regimes, no_lags, beta_hat=None):
    """
    :param number_regimes: Number of regimes in the model
    :param no_lags:  Number of lags in the VECM estimates
    :param beta_hat: Default None, uses journalese procedure to tabulate
    :return:
    delta_y_t: endogenous variable
    z_t_1: right-hand variable
    ols_resid:  residuals from the initial OLS estimates of the VECM model
    """

    delta_y_t, z_t_1, ols_resid = data.data_matrix(data.df, no_lags, beta_hat)
    k, obs = delta_y_t.shape

    # temp log squared residuals array to save values
    u_u = np.zeros([k*k, obs])
    # tabulate log squared residuals using the residuals
    for t in range(obs):
        u = np.log(ols_resid[:, t])
        u_u[:, t] = np.repeat(u, k) + np.tile(u, k)
    log_vector_b = 0.5 * logsumexp(u_u) - np.log(obs)  # initial b matrix vectorized
    vector_b = np.exp(log_vector_b) + np.random.normal(0, 0.1, size=(k * k, 1))

    int_params = {'log_trans_prob_matrix': np.log(np.ones([number_regimes, number_regimes]) / number_regimes),
                  'epsilon_0': np.log(np.ones(number_regimes) / number_regimes),
                  'B_matrix': vec_matrix(vector_b)
                  }

    return int_params, delta_y_t, z_t_1


# observations


# expectation step
# creating empty array to store tabulated values of eta_t(regimes,obs)
eta = np.zeros([regimes, obs])
