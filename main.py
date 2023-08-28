import numpy as np
from scipy.special import logsumexp
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


def initialize(number_regimes, lags, beta=None):
    delta_y_t, z_t_1, ols_resid = data.data_matrix(data.df, lags, beta)
    log_resid = np.log(ols_resid)
    k, obs = delta_y_t.shape
    # matrix to save initial log(b) values
    log_b = np.ones([k,k])
    log_b[np.diag_indices_from(log_b)] = logsumexp(2*log_resid, axis=1) # replacing diagonal elements

    int_params = {'log_trans_prob_matrix': np.log(np.ones([number_regimes, number_regimes]) / number_regimes),
                  'epsilon_0': np.log(np.ones(number_regimes) / number_regimes),
                  'B_matrix':

                }
    return int_params, delta_y_t, z_t_1


# observations


np.log(ols_resid)






# expectation step
# creating empty array to store tabulated values of eta_t(regimes,obs)
eta = np.zeros([regimes, obs])


