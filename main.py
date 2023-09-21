import numpy as np
from scipy import stats
from scipy.linalg import pinv
from scipy.linalg import sqrtm
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.covariance import LedoitWolf

import data as data
import optimization as op
from matrix_operations import vec_matrix, mat_vec, replace_diagonal

print(f'this is eps:{np.finfo(np.float64).eps}')
# Model
regimes = 3
lags = 3
beta = np.array([0, 0, 0, 1]).reshape((-1, 1))  # must be shaped according to (number of variables, 1)
np.random.seed(42)


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
    b_matrix = b_matrix + np.random.normal(0, 0.01, size=(k, k))
    lam = replace_diagonal(np.random.normal(1, 0, size=k))
    print(lam)
    sigma_array = np.zeros([regimes, k, k])
    for regime in range(regimes):
        if regime == 0:
            sigma_array[regime, :, :] = b_matrix @ b_matrix.T
        else:
            sigma_array[regime, :, :] = b_matrix @ lam @ b_matrix.T

    params = {'regimes': number_regimes,
              'epsilon_0': (np.log(np.ones(number_regimes) / number_regimes)).reshape(-1, 1),
              'transition_prob_mat': np.log(np.ones([number_regimes, number_regimes]) / number_regimes),
              'B_matrix': b_matrix,
              'lambda_m': np.identity(b_matrix.shape[0]),
              'sigma': sigma_array,
              'residuals': ols_resid,
              'VECM_params': None}
    return params, delta_y_t, z_t_1,


# expectation step
def e_step(params):
    k, obs = params['residuals'].shape  # number of observations

    # arrays to save estimated values NOTE THAT ALL PROBABILITIES ARE IN LOG!
    conditional_prob = np.zeros([params['regimes'], obs])  # y_t|s_t = j conditional density of Y for a given state
    filtered_prob = np.zeros(conditional_prob.shape)  # s_t|y_t
    predicted_prob = np.zeros(conditional_prob.shape)  # s_t | y_t-1
    likelihoods = np.zeros([obs])  # marginal density of observed variables
    smoothed_joint_prob = np.zeros([obs, params['regimes'], params['regimes']])  # joint prob S_t = j, s_t-1 = k | y_T
    smoothed_prob = np.zeros(conditional_prob.shape)  # prob S_t = j| y_T
    t_00_smth_jnt_prob = np.zeros([1, params['regimes'], params['regimes']])
    t_00_smoothed_prob = np.zeros([params['regimes'], 1])
    vec_trans_prob_mat = mat_vec(params['transition_prob_mat'])
    # conditional_prob of the observed variables IN LOG
    for r in range(params['regimes']):
        conditional_prob[r, :] = stats.multivariate_normal(mean=None,
                                                           cov=params['sigma'][r, :, :]).logpdf(params['residuals'].T).T

    # Hamilton filter/ filtered probability IN LOG
    for t_ in range(0, obs):

        # taking starting values
        if t_ == 0:
            flt_prob_temp = params['epsilon_0']
        else:
            flt_prob_temp = filtered_prob[:, [t_ - 1]]

        # Predicted prob.
        start = 0
        values = vec_trans_prob_mat + np.tile(flt_prob_temp.T, params['regimes']).reshape(-1, 1)

        for r in range(params['regimes']):
            sum_over = values[start:start + params['regimes']]
            predicted_prob[r, t_] = logsumexp(sum_over)
            start += params['regimes']

        # joint prob
        p_st_yt = filtered_prob[:, t_] + conditional_prob[:, t_]

        # log likelihood of each observation
        likelihoods[t_] = logsumexp(p_st_yt)  #

        # filtered Prob
        filtered_prob[:, t_] = p_st_yt - likelihoods[t_]
    print(f'filtered prob: {filtered_prob}')
    # smooth prob
    smoothed_prob[:, [-1]] = filtered_prob[:, [-1]]

    for t_ in range(obs - 2, -2, -1):  # T-1, ..., 1
        for r_j in range(params['regimes']):  # regime j at time t
            for r_k in range(params['regimes']):  # regime k at time t+1
                tem_smoothed_joint_prob = smoothed_prob[r_k, t_ + 1] \
                                          + filtered_prob[r_j, t_] \
                                          + params['transition_prob_mat'][r_j, r_k] \
                                          - predicted_prob[r_k, t_ + 1]
                if t_ >= 0:
                    smoothed_joint_prob[t_, r_j, r_k] = tem_smoothed_joint_prob
                else:
                    t_00_smth_jnt_prob[0, r_j, r_k] = tem_smoothed_joint_prob

        if t_ >= 0:
            for r_j in range(params['regimes']):
                smoothed_prob[r_j, t_] = logsumexp(smoothed_joint_prob[t_, r_j, :])
        else:
            for r_j in range(params['regimes']):
                t_00_smoothed_prob[r_j, 0] = logsumexp(t_00_smth_jnt_prob[0, r_j, :])

    return smoothed_joint_prob, smoothed_prob, t_00_smoothed_prob, likelihoods.sum()


def m_step(smoothed_joint_prob, smoothed_prob, x0, zt, delta_y, parameters):
    # optimization additional arguments (tuple)
    k = parameters['residuals'].shape[0]
    ####################################
    # estimating transition probability
    ####################################

    # transition probability matrix is in logs

    transition_prob_mat = logsumexp(smoothed_joint_prob[0:-2, :, :], axis=0) - logsumexp(smoothed_prob[:, 0:-2], axis=1)

    print(f'this is transition prob mat:{transition_prob_mat} ')


    ####################################
    # estimating Covariance matrices
    ####################################

    # bounds to ensure positive semi-definite
    bound_list = []
    for i in range(len(x0)):
        if i < k ** 2:
            bound_list.append((None, None))
        else:
            bound_list.append((0.01, None))
    bound_list = tuple(bound_list)

    # no need to take  exponential of smoothed prob as optimization file does it there.
    op_params = [parameters['regimes'], k, parameters['residuals'], smoothed_prob]

    res = minimize(op.marginal_density, x0, args=op_params, method='cobyla',
                   bounds=bound_list)
    print(res.message)
    b_mat, sigma = op.b_matrix_sigma(res.x, k, parameters['regimes'])

    mean_zero = np.zeros([k])
    for regime in range(parameters['regimes']):
        x = np.random.multivariate_normal(mean=mean_zero, cov=sigma[regime, :, :], size=50, check_valid='ignore')
        cov = LedoitWolf().fit(x)
        sigma[regime, :, :] = cov.covariance_

    print(f'this is b mat :{b_mat}')
    print(f'this is sigma:{sigma}')

    ####################################
    # estimate weighted least-square parameters
    ####################################

    for regime in range(parameters['regimes']):
        t_sum = np.zeros([zt.shape[0], zt.shape[0]])
        m_sum = np.zeros([zt.shape[0] * k, zt.shape[0] * k])
        m_sum_numo = np.zeros([zt.shape[0] * k, k])
        t_sum_numo = np.zeros([zt.shape[0] * k, 1])

        for t in range(zt.shape[1]):
            t_sum += np.exp(smoothed_prob[regime, t]) * zt[:, [t]] @ zt[:, [t]].T
        m_sum += np.kron(t_sum, pinv(sigma[regime, :, :]))
        denominator = pinv(m_sum)

    for t in range(zt.shape[1]):
        for regime in range(parameters['regimes']):
            m_sum_numo += np.kron(np.exp(smoothed_prob[regime, t]) * zt[:, [t]], pinv(sigma[regime, :, :]))
        t_sum_numo += m_sum_numo @ delta_y[:, [t]]

    theta_hat = denominator @ t_sum_numo

    ####################################
    # residuals estimate
    ####################################

    resid = np.zeros(delta_y.shape)
    for t in range(zt.shape[1]):
        resid[:, [t]] = delta_y[:, [t]] - np.kron(zt[:, [t]].T, np.identity(delta_y.shape[0])) @ theta_hat

    return transition_prob_mat, b_mat, sigma, theta_hat, resid, res.x


# initialization


def run_em(regimes, lags, beta_hat=beta, max_itr=100):
    x0 = [5.96082486e+01, 5.74334765e-01, 2.83277325e-01, 3.66479528e+00,
          -2.08881529e-01, 6.32170541e-04, -1.09137417e-01, -3.80763529e-01,
          4.24379418e+00, 1.83658083e-01, 2.16692718e-03, 1.29590368e+00,
          2.20826553e+00, -2.98484217e-01, -5.38269363e-03, 1.19668239e-03,
          0.012, 0.102, 0.843, 16.52]

    params, delta_y_t, z_t_1 = initialize(regimes, lags, beta_hat)
    print('initial expectation step')
    smoothed_joint_prob, smoothed_prob, params['epsilon_0'], log_likelihoods = e_step(params)
    while True:
        break

    avg_loglikelihoods = []
    i = 0
    while i < max_itr:
        avg_loglikelihood = log_likelihoods
        avg_loglikelihoods.append(avg_loglikelihood)
        if len(avg_loglikelihoods) > 2 and abs(avg_loglikelihoods[-1] - avg_loglikelihoods[-2]) < 0.0001:
            print(f'optimization successfully terminated reached a solution in {i} iterations!')
            break

        # maximization
        params['transition_prob_mat'], \
            params['B_matrix'], \
            params['sigma'], \
            params['VECM_params'], \
            params['residuals'], x0 = m_step(smoothed_joint_prob, smoothed_prob, x0, z_t_1, delta_y_t, params)

        smoothed_joint_prob, smoothed_prob, params['epsilon_0'], log_likelihoods = e_step(params)
        print('============================================')
        print(f'smoothed prob: {np.exp(smoothed_prob)} ')
        print(f"transition prob matrix : {np.exp(params['transition_prob_mat'])}")
        print('============================================')


        i += 1
        if i == max_itr:
            print('maximum iteration reached!')

    return params


print(run_em(regimes, lags, beta_hat=beta, max_itr=15))
