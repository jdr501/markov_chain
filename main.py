import numpy as np
from scipy.special import logsumexp
from matrix_operations import vec_matrix, mat_vec, log_space_product
from scipy import stats
from scipy.linalg import sqrtm
import data as data
import optimization as op
from scipy.optimize import minimize
from scipy.linalg import pinv
from sklearn.covariance import LedoitWolf


# Model
regimes = 2
lags = 2
beta = np.array([0, 0, 0, 1]).reshape((-1, 1))  # must be shaped according to (number of variables, 1)


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

    # temp array to save u u.T values
    u_u = np.zeros([k * k, obs])
    # tabulate log squared residuals using the residuals
    for t in range(obs):
        u = ols_resid[:, t]
        u_u[:, t] = np.repeat(u, k) * np.tile(u, k)
    b_matrix = sqrtm(vec_matrix(u_u.sum(axis=1) / obs))
    b_matrix = b_matrix + np.random.normal(0, 0.1, size=(k, k))
    lam = []
    sigma_array = np.zeros([regimes, k, k])
    for regime in range(regimes):
        sigma_array[regime, :, :] = b_matrix @ b_matrix.T

    ini_params = {'log_trans_prob_matrix': np.log(np.ones([number_regimes, number_regimes]) / number_regimes),
                  'log_epsilon_0': np.log(np.ones(number_regimes) / number_regimes),
                  'B_matrix': b_matrix,
                  'Sigma': sigma_array
                  }
    return ini_params, delta_y_t, z_t_1, ols_resid


# expectation step
def e_step(residuals, params, regimes, epsilon_t0=None):
    obs = residuals.shape[1]  # number of observations
    vec_log_p_trans = mat_vec(params['log_trans_prob_matrix'].T)
    ## making sure sigma is positive semidefenite


    # arrays to save filtered and forecast probabilities and likelihoods
    epsilon_t_t = np.zeros([regimes, obs])
    epsilon_t_t_1 = np.zeros([regimes, obs])
    likelihoods = np.zeros([obs])

    # conditional densities of the observed
    log_eta_t = np.zeros([regimes, obs])
    # following is the log(log_eta_t)
    for regime in range(regimes):
        log_eta_t[regime, :] = stats.multivariate_normal(mean=None, cov=params['Sigma'][regime, :, :]).logpdf(residuals.T).T
    # vectorized log transition probabilities
    vec_log_b = mat_vec(params['log_trans_prob_matrix'])

    for t in range(obs):
        # making sure that starting filtered values are taken properly
        if t == 0 and epsilon_t0 is None:
            filtered_prob = params['log_epsilon_0']

        elif t == 0 and epsilon_t0 is not None:
            filtered_prob = epsilon_t0
        else:
            filtered_prob = epsilon_t_t[:, t]

        filtered_prob = filtered_prob.reshape([-1, 1])

        # tabulating the epsilon_t_t-1
        start = 0
        values = vec_log_b + np.tile(filtered_prob.T, regimes).T

        for regime in range(regimes):
            epsilon_t_t_1[regime, t] = logsumexp(values[start:start + regimes - 1])
            start += regimes

        p_st_yt = epsilon_t_t[:, t] + log_eta_t[:, t]
        likelihoods[t] = logsumexp(p_st_yt)
        epsilon_t_t[:, t] = p_st_yt - likelihoods[t]

    # smooth prob saving array
    smth_prob = np.zeros([regimes, obs])
    middle_array = np.zeros([regimes, obs])
    smth_joint_prob = np.zeros([regimes * regimes, obs])  # joint prob of two different states
    # smoothed prob. iteration
    smth_prob[:, -1] = epsilon_t_t[:, -1]
    # this is starting from T-1 to index zero.we have T-2 because we already got T from filtered prob
    for t in range(obs - 2, -1, -1):
        middle_array[:, [t]] = smth_prob[:, [t + 1]] - log_space_product(params['log_trans_prob_matrix'],
                                                                         epsilon_t_t[:, [t]])
        smth_prob[:, [t]] = log_space_product(params['log_trans_prob_matrix'].T, middle_array[:, [t]]) + \
                            epsilon_t_t[:, [t]]

    for t in range(obs - 1):
        middle_array[:, [t]] = smth_prob[:, [t + 1]] - log_space_product(params['log_trans_prob_matrix'],
                                                                         epsilon_t_t[:, [t]])
        smth_joint_prob[:, [t]] = np.repeat(middle_array[:, [t]], regimes).reshape(-1, 1) + \
                                  np.tile(epsilon_t_t[:, [t]], regimes).reshape(-1, 1)

        smth_joint_prob[:, [t]] = vec_log_p_trans + smth_joint_prob[:, [t]]
    return epsilon_t_t, likelihoods.sum(), smth_prob, smth_joint_prob


def m_step(smth_joint_prob, smth_prob, no_regimes, parameters, x0, zt, delta_y):
    # estimating transition probability
    log_vec_p = logsumexp(smth_joint_prob, axis=1) - np.tile(logsumexp(smth_prob, axis=1), no_regimes)

    # estimating Covariance matrices
    bound_list = []
    for i in range(len(x0)):
        if i < parameters[1] ** 2:
            bound_list.append((None, None))
        else:
            bound_list.append((0.01, None))

    bound_list = tuple(bound_list)

    res = minimize(op.marginal_density, x0, args=parameters, method='cobyla',
                   bounds=bound_list)
    print(res.message)
    b_mat, sigma = op.b_matrix_sigma(res.x, parameters[1], regimes)

    mean_zero = np.zeros(parameters[1])
    for regime in range(regimes):
        x = np.random.multivariate_normal(mean=mean_zero, cov=sigma[regime, :, :], size=50, check_valid='ignore')
        cov = LedoitWolf().fit(x)
        sigma[regime, :, :] = cov.covariance_



    # estimate weighted  least square parameters
    for regime in range(regimes):
        t_sum = np.zeros([zt.shape[0], zt.shape[0]])
        m_sum = np.zeros([zt.shape[0]*parameters[1], zt.shape[0]*parameters[1]])
        m_sum_numo = np.zeros([zt.shape[0]*parameters[1], parameters[1]])
        t_sum_numo = np.zeros([zt.shape[0] * parameters[1], 1])
        for t in range(zt.shape[1]):
            t_sum += np.exp(smth_prob[regime, t]) * zt[:, [t]] @ zt[:, [t]].T
        m_sum += np.kron(t_sum,   pinv(sigma[regime, :, :]))
        denominator = pinv(m_sum)
    for t in range(zt.shape[1]):
        for regime in range(regimes):
            m_sum_numo += np.kron(np.exp(smth_prob[regime, t]) * zt[:, [t]], pinv(sigma[regime, :, :]))
        t_sum_numo += m_sum_numo @ delta_y[:, [t]]

    theta_hat = denominator @ t_sum_numo

    #residuals estimate
    resid = np.zeros(delta_y.shape)
    for t in range(zt.shape[1]):
        resid[:, [t]] = delta_y[:, [t]] - np.kron(zt[:, [t]].T, np.identity(delta_y.shape[0])) @ theta_hat

    return log_vec_p, b_mat, sigma, theta_hat, resid, res.x


# initialization


def run_em(regimes,lags,beta_hat=beta):
    x0 = [5.96082486e+01, 5.74334765e-01, 2.83277325e-01, 3.66479528e+00,
          -2.08881529e-01, 6.32170541e-04, -1.09137417e-01, -3.80763529e-01,
          4.24379418e+00, 1.83658083e-01, 2.16692718e-03, 1.29590368e+00,
          2.20826553e+00, -2.98484217e-01, -5.38269363e-03, 1.19668239e-03,
          0.012, 0.102, 0.843, 16.52]
    params_expectation, delta_y, zt, resid = initialize(regimes, lags, beta_hat)
    epsilon_t_t, likelihoods, smth_prob, smth_joint_prob = e_step(resid, params_expectation, regimes, epsilon_t0=None)
    avg_loglikelihoods = []
    while True:
        avg_loglikelihood = np.mean(likelihoods)
        avg_loglikelihoods.append(avg_loglikelihood)
        if len(avg_loglikelihoods) > 2 and abs(avg_loglikelihoods[-1] - avg_loglikelihoods[-2]) < 0.0001:
            break

        # maximization

        parameters = [regimes, 4, resid, smth_prob]

        log_vec_p, b_mat, sigma, theta_hat, resid, x0 = m_step(smth_joint_prob, smth_prob, regimes, parameters, x0, zt, delta_y)

        params_expectation = {'log_trans_prob_matrix': vec_matrix(log_vec_p),
                          'log_epsilon_0': smth_prob[:, [0]],
                          'B_matrix': b_mat,
                          'Sigma': sigma
                          }
        #expectation
        epsilon_t_t, likelihoods, smth_prob, smth_joint_prob = e_step(resid, params_expectation, regimes, epsilon_t0=None)

run_em(regimes, lags, beta_hat=beta)