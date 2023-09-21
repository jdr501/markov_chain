import pandas as pd
import numpy as np


def logsumexp(sum):
    max = np.max(sum)
    if max > 1e30 or max < 1e-30:
        return max
    else:
        a = np.exp(sum[0] - max)
        for i in range(1, len(sum)):
            a += np.exp(sum[i] - max)
        return np.log(a) + max


def forward(trans_prob, ln_eta_t, ini_dist):
    alpha = np.zeros((ln_eta_t.shape[1], trans_prob.shape[0]))
    alpha[0, :] = ini_dist * ln_eta_t[:, 0]

    for t in range(1, ln_eta_t.shape[1]):
        for j in range(trans_prob.shape[0]):
            # Matrix Computation Steps
            #                  ((1x2) . (1x2))      *     (1)
            #                        (1)            *     (1)
            alpha_trans = alpha[t - 1] + trans_prob[:, j]

            alpha[t, j] = logsumexp(alpha_trans) + ln_eta_t[j, t]

    return np.exp(alpha)


def backward(V, trans_prob, b):
    beta = np.zeros((V.shape[0], trans_prob.shape[0]))

    # setting beta(T) = 1
    beta[V.shape[0] - 1] = np.ones((trans_prob.shape[0]))

    # Loop in backward way from T-1 to
    # Due to python indexing the actual loop will be T-2 to 0
    for t in range(V.shape[0] - 2, -1, -1):
        for j in range(trans_prob.shape[0]):
            beta[t, j] = (beta[t + 1] * b[:, V[t + 1]]).dot(trans_prob[j, :])

    return beta


def baum_welch(V, a, b, initial_distribution, n_iter=100):
    M = a.shape[0]
    T = len(V)

    for n in range(n_iter):
        alpha = forward(V, a, b, initial_distribution)
        beta = backward(V, a, b)

        xi = np.zeros((M, M, T - 1))
        for t in range(T - 1):
            denominator = np.dot(np.dot(alpha[t, :].T, a) * b[:, V[t + 1]].T, beta[t + 1, :])
            for i in range(M):
                numerator = alpha[t, i] * a[i, :] * b[:, V[t + 1]].T * beta[t + 1, :].T
                xi[i, :, t] = numerator / denominator

        gamma = np.sum(xi, axis=1)
        a = np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))

        # Add additional T'th element in gamma
        gamma = np.hstack((gamma, np.sum(xi[:, :, T - 2], axis=0).reshape((-1, 1))))

        K = b.shape[1]
        denominator = np.sum(gamma, axis=1)
        for l in range(K):
            b[:, l] = np.sum(gamma[:, V == l], axis=1)

        b = np.divide(b, denominator.reshape((-1, 1)))

    return {"a": a, "b": b}


data = pd.read_csv('data_python.csv')

V = data['Visible'].values

# Transition Probabilities
a = np.ones((2, 2))
a = a / np.sum(a, axis=1)

# Emission Probabilities
b = np.array(((1, 3, 5), (2, 4, 6)))
b = b / np.sum(b, axis=1).reshape((-1, 1))

# Equal Probabilities for the initial distribution
initial_distribution = np.array((0.5, 0.5))

print(baum_welch(V, a, b, initial_distribution, n_iter=100))
