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
params['regimes'] = 2
lags = 3
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
    sigma_array = np.zeros([params['regimes'], k, k])
    for regime in range(params['regimes']):
        sigma_array[regime, :, :] = b_matrix @ b_matrix.T

    ini_params = {'log_trans_prob_matrix': np.log(np.ones([number_regimes, number_regimes]) / number_regimes),
                  'log_epsilon_0': np.log(np.ones(number_regimes) / number_regimes),
                  'B_matrix': b_matrix,
                  'Sigma': sigma_array
                  }
    return ini_params, delta_y_t, z_t_1, ols_resid


parameters = {'regimes': params['regimes'],
              'epsilon_0': epsilon_0,
              'transition_prob_mat': p_mat,
              'B_matrix': b_mat,
              'lambda_m': lam_m,
              'sigma': sigma_array,
              'VECM_params': theta_hat,
              'residuals': residuals}


# expectation step
def e_step(residuals, params):
    k, obs = residuals.shape  # number of observations

    # arrays to save estimated values NOTE THAT ALL PROBABILITIES ARE IN LOG!
    conditional_prob = np.zeros([params['regimes'], obs])  # y_t|s_t = j conditional density of Y for a given state
    filtered_prob = np.zeros(conditional_prob.shape)  # s_t|y_t
    predicted_prob = np.zeros(conditional_prob.shape)  # s_t | y_t-1
    likelihoods = np.zeros([obs])  # marginal density of observed variables
    smoothed_joint_prob = np.zeros([obs, params['regimes'], params['regimes']])  # joint prob S_t = j, s_t-1 = k | y_T
    smoothed_prob = np.zeros(conditional_prob.shape)  # prob S_t = j| y_T
    vec_trans_prob_mat = mat_vec(params['B_matrix'].T)

    # conditional_prob of the observed variables IN LOG
    for r in range(params['regimes']):
        conditional_prob[r, :] = stats.multivariate_normal(mean=None,
                                                           cov=params['sigma'][r, :, :]).logpdf(residuals.T).T

    # Hamilton filter/ filtered probability IN LOG

    for t_ in range(obs):

        # taking starting values
        if t_ == 0:
            flt_prob_temp = params['epsilon_0']
        else:
            flt_prob_temp = filtered_prob[:, [t_]]

        # Predicted prob.
        start = 0
        values = vec_trans_prob_mat + np.tile(flt_prob_temp.T, params['regimes']).T
        for r in range(params['regimes']):
            predicted_prob[r, t_] = logsumexp(values[start:start + params['regimes'] - 1])
            start += params['regimes']
        # joint prob
        p_st_yt = filtered_prob[:, t_] + conditional_prob[:, t_]

        # log likelihood of each observation
        likelihoods[t_] = logsumexp(p_st_yt)

        # filtered Prob
        filtered_prob[:, t_] = p_st_yt - likelihoods[t_]

    # smooth prob
    for t_ in range(obs - 1, -1, -1):  # from T ... 0
        if t_ == obs - 1:
            smoothed_prob[:, [t_]] = filtered_prob[:, [t_]]
        else:
            for r_j in range(params['regimes']):
                smoothed_prob[r_j, t_] = logsumexp(smoothed_joint_prob[t_, r_j, :])
        for r_j in range(params['regimes']):  # regime j at time t
            for r_k in range(params['regimes']):  # regime k at time t+1
                smoothed_joint_prob[t_, r_j, r_k] = smoothed_prob[r_k, t_ + 1] \
                                                    + filtered_prob[r_j, t_] \
                                                    + params['B_matrix'][r_j, r_k] - predicted_prob[r_k, t_ + 1]

    return likelihoods.sum(), smoothed_prob, smoothed_joint_prob


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
    print(res.x)

    b_mat, sigma = op.b_matrix_sigma(res.x, parameters[1], params['regimes'])
    """ 
    mean_zero = np.zeros(parameters[1])
    for regime in range(regimes):
        x = np.random.multivariate_normal(mean=mean_zero, cov=sigma[regime, :, :], size=50, check_valid='ignore')
        cov = LedoitWolf().fit(x)
        sigma[regime, :, :] = cov.covariance_
    """
    # estimate weighted  least square parameters
    for regime in range(params['regimes']):
        t_sum = np.zeros([zt.shape[0], zt.shape[0]])
        m_sum = np.zeros([zt.shape[0] * parameters[1], zt.shape[0] * parameters[1]])
        m_sum_numo = np.zeros([zt.shape[0] * parameters[1], parameters[1]])
        t_sum_numo = np.zeros([zt.shape[0] * parameters[1], 1])
        for t in range(zt.shape[1]):
            t_sum += np.exp(smth_prob[regime, t]) * zt[:, [t]] @ zt[:, [t]].T
        m_sum += np.kron(t_sum, pinv(sigma[regime, :, :]))
        denominator = pinv(m_sum)
    for t in range(zt.shape[1]):
        for regime in range(params['regimes']):
            m_sum_numo += np.kron(np.exp(smth_prob[regime, t]) * zt[:, [t]], pinv(sigma[regime, :, :]))
        t_sum_numo += m_sum_numo @ delta_y[:, [t]]

    theta_hat = denominator @ t_sum_numo

    # residuals estimate
    resid = np.zeros(delta_y.shape)
    for t in range(zt.shape[1]):
        resid[:, [t]] = delta_y[:, [t]] - np.kron(zt[:, [t]].T, np.identity(delta_y.shape[0])) @ theta_hat

    return log_vec_p, b_mat, sigma, theta_hat, resid, res.x


# initialization


def run_em(regimes, lags, beta_hat=beta, max_itr=100):
    x0 = [5.96082486e+01, 5.74334765e-01, 2.83277325e-01, 3.66479528e+00,
          -2.08881529e-01, 6.32170541e-04, -1.09137417e-01, -3.80763529e-01,
          4.24379418e+00, 1.83658083e-01, 2.16692718e-03, 1.29590368e+00,
          2.20826553e+00, -2.98484217e-01, -5.38269363e-03, 1.19668239e-03,
          0.012, 0.102, 0.843, 16.52]
    params_expectation, delta_y, zt, resid = initialize(regimes, lags, beta_hat)
    epsilon_t_t, likelihoods, smth_prob, smth_joint_prob = e_step(resid, params_expectation, regimes, epsilon_t0=None)
    avg_loglikelihoods = []
    i = 0
    while i < 3:
        avg_loglikelihood = np.mean(likelihoods)
        avg_loglikelihoods.append(avg_loglikelihood)
        if len(avg_loglikelihoods) > 2 and abs(avg_loglikelihoods[-1] - avg_loglikelihoods[-2]) < 0.0001:
            break

        # maximization

        parameters = [regimes, 4, resid, smth_prob]

        log_vec_p, b_mat, sigma, theta_hat, resid, x0 = m_step(smth_joint_prob, smth_prob, regimes, parameters, x0, zt,
                                                               delta_y)

        params_expectation = {'log_trans_prob_matrix': vec_matrix(log_vec_p).T,
                              'log_epsilon_0': smth_prob[:, [0]],
                              'B_matrix': b_mat,
                              'Sigma': sigma
                              }
        # expectation
        print(b_mat)
        print(sigma)
        epsilon_t_t, likelihoods, smth_prob, smth_joint_prob = e_step(resid, params_expectation, regimes,
                                                                      epsilon_t0=None)
        i += 1

    return params_expectation, smth_prob, i


print(run_em(params['regimes'], lags, beta_hat=beta, ))
