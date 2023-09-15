import numpy as np
from scipy import stats
from scipy.linalg import pinv
from scipy.linalg import sqrtm
from scipy.optimize import minimize
# from scipy.special import logsumexp
from sklearn.covariance import LedoitWolf

import data as data
import optimization as op
from matrix_operations import vec_matrix, mat_vec, replace_diagonal

# print(np.log(0))
# Model
regimes = 2
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

def logsumexp(sum):
    max = np.max(sum)
    if max > 1e30 or max < -1e30:
        return max
    else:
        a = np.exp(sum[0] - max)
        for i in range(1, len(sum)):
            a += np.exp(sum[i] - max)
        return np.log(a) + max


def cond_prob(param):
    obs = param['residuals'].shape[1]
    conditional_prob = np.zeros([param['regimes'], obs])  # y_t|s_t = j conditional density of Y for a given state
    for r in range(param['regimes']):
        conditional_prob[r, :] = stats.multivariate_normal(mean=None,
                                                           cov=param['sigma'][r, :, :]).logpdf(param['residuals'].T).T
    print(f'shape of eta t :{conditional_prob.shape} ')
    return conditional_prob


def forward(trans_prob, ln_eta_t, ini_dist):
    alpha = np.zeros((trans_prob.shape[0], ln_eta_t.shape[1]))
    ln_epsilon_t = np.zeros((trans_prob.shape[0], ln_eta_t.shape[1]))
    sum_alpha_t = np.zeros(alpha.shape[1])
    alpha[:, [0]] = ini_dist * ln_eta_t[:, [0]]
    print(alpha[:, [0]])
    for t in range(1, ln_eta_t.shape[1]):
        for j in range(trans_prob.shape[0]):
            # Matrix Computation Steps
            #                  ((1x2) . (1x2))      *     (1)
            #                        (1)            *     (1)
            alpha_trans = alpha[:, t - 1] + trans_prob[:, j]
            alpha[j, t] = logsumexp(alpha_trans) + ln_eta_t[j, t]
    for t in range(0, ln_eta_t.shape[1]):
        sum_alpha_t[t] = logsumexp(alpha[:, t])
        ln_epsilon_t[:, [t]] = alpha[:, [t]] - sum_alpha_t[t]

    return ln_epsilon_t, alpha


def backward(trans_prob, ln_eta_t):
    back = np.zeros((trans_prob.shape[0], ln_eta_t.shape[1]))
    scaled_back = np.zeros((trans_prob.shape[0], ln_eta_t.shape[1]))
    sum_back_t = np.zeros(back.shape[1])

    # setting beta(T) = 1
    back[:, [ln_eta_t.shape[1] - 1]] = np.ones([trans_prob.shape[0], 1])

    # Loop in backward way from T-1 to
    # Due to python indexing the actual loop will be T-2 to 0
    for t in range(ln_eta_t.shape[1] - 2, -1, -1):
        for j in range(trans_prob.shape[0]):
            back_trans = ln_eta_t[j, [t + 1]] + trans_prob[j, :]
            back[j, t] = back[j, t + 1] + logsumexp(back_trans)

    for t in range(0, ln_eta_t.shape[1]):
        sum_back_t[t] = logsumexp(back[:, t])
        scaled_back[:, [t]] = back[:, [t]] - sum_back_t[t]
    print(scaled_back)
    return scaled_back, back

def smoothed(forward_prob, scaled_back):
    smoothed_prob = np.zeros(forward_prob.shape)
    for t in range(forward_prob.shape[1]):
        for j in range(forward_prob.shape[0]):
            smoothed_prob[j, t] = forward_prob[j, t] * scaled_back[j, t]
        smoothed_sum = logsumexp(smoothed_prob[:, t])
        smoothed_prob[:, [t]] = smoothed_prob[:, [t]] - smoothed_sum

    return smoothed_prob


def trans_prob_mat(trans_prob, eta_t, param, n_iter=1):
    trans_prob = np.exp(trans_prob)
    eta_t = np.exp(eta_t).T
    m = trans_prob.shape[0]
    obs = eta_t.shape[1]

    for n in range(n_iter):
        ln_epsilon_t, alpha = forward(param['transition_prob_mat'], np.log(eta_t.T), param['epsilon_0'])
        scaled_back, beta = backward(param['transition_prob_mat'], np.log(eta_t.T))
        alpha = np.exp(alpha).T
        beta = np.exp(beta).T

        xi = np.zeros([m, m, obs - 1])
        for t in range(obs - 1):

            denominator = np.dot(np.dot(alpha[t, :].T, trans_prob) * eta_t[[t + 1],:].T, beta[t + 1, :])
            for i in range(m):
                numerator = alpha[t, i] * trans_prob[i, :] * eta_t[[t + 1], i].T * beta[t + 1, :].T

                xi[i, :, t] = numerator / denominator

        gamma = np.sum(xi, axis=1)
        trans_prob = np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))

    return np.log(trans_prob)



# expectation step
def e_step(params):
    eta_t = cond_prob(params)
    print(f'shape of eta t{eta_t.shape}')
    ln_epsilon_t, alpha_forward = forward(params['transition_prob_mat'], eta_t, params['epsilon_0'])
    scaled_back, beta_back = backward(params['transition_prob_mat'], eta_t)
    smoothed_prob = smoothed(ln_epsilon_t, scaled_back)
    t_00_smoothed_prob = smoothed_prob[:, [0]]

    return eta_t, smoothed_prob, t_00_smoothed_prob, None


def m_step(eta_t, smoothed_prob, x0, zt, delta_y, parameters):
    # optimization additional arguments (tuple)
    print('========Optimization=========')
    k, obs = parameters['residuals'].shape
    trans_prob = parameters['transition_prob_mat']

    
    ####################################
    # estimating transition probability
    ####################################
    transition_prob_mat = trans_prob_mat(trans_prob, eta_t, parameters, n_iter=1)

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
    print(f'this is itteration: {i}//////////')
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
