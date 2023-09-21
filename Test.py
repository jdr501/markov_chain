import math
import numpy as np
import statsmodels.api as sm
from statsmodels.base.model import GenericLikelihoodModel
from scipy.stats import poisson
#from patsy import dmatrices
import statsmodels.graphics.tsaplots as tsa
from matplotlib import pyplot as plt
from statsmodels.tools.numdiff import approx_hess1, approx_hess2, approx_hess3

class MultivariateHMM(GenericLikelihoodModel):
    def __init__(self,endog, exog,  k_regimes= 2,
                 loglike =None,
                 score = None,
                 hessian = None,
                 missing = 'none',
                 extra_params_names = None,
                 **kwds):
        super(MultivariateHMM,self).__init__(endo = endog,
                                             exo=exog,
                                             loglike=loglike, score=score,
                                             hessian=hessian,
                                             missing=missing,
                                             extra_params_names=extra_params_names,
                                             kwds= kwds)
        self.y = np.array(self.endog)
        self.k_regimes = k_regimes


        # exog.shape[0] size matrix of regression coefficients
        self.beta_matrix = np.ones([self.exog.shape[0],1])

        # k x k matrix of psuedo transition probabilities which can range from -inf to +inf during
        # optimization. Initialized to 1.0/k
        self.q_matrix = np.ones([self.k_regimes, self.k_regimes]) * (1.0 / self.k_regimes)
        print('self.q_matrix=' + str(self.q_matrix))

        # The regime wise matrix of vector Gaussian means. These would be updated during the optimization
        # loop
        self.mu_matrix = []

        # k x k matrix of the real Markov transition probabilities which will be calculated from
        # the q-matrix using a standardization technique. Initialized to 1.0/k
        self.gamma_matrix = np.ones([self.k_regimes, self.k_regimes]) * (1.0 / self.k_regimes)
        print('self.gamma_matrix=' + str(self.gamma_matrix))

        # The Markov state probabilities. Also referred to as pi. but we'll use delta since pi is
        # often used to refer to the mean
        self.delta_matrix = np.ones([self.k_regimes, self.exog.shape[1]]) * (1.0 / self.k_regimes)
        print('self.delta_matrix=' + str(self.delta_matrix))


        # The vector of initial values for all the parameters, beta and q, B_matrix and lambdas that the optimizer will
        # optimize
        self.start_params = np.ones(self.exog.shape[0])
        self.start_params = np.append(self.start_params, self.q_matrix.flatten())
        b_matrix = (np.ones([self.endog.shape[0],self.endog.shape[0]])/2).flatten()
        self.start_params = np.append(self.start_params, b_matrix)
        lambda_array= np.ones([self.endog.shape[0],self.k_regimes]).flatten()
        self.start_params = np.append(self.start_params, lambda_array)




        print('self.start_params=' + str(self.start_params))
        # A very tiny number (machine specific). Used by the LL function.
        self.EPS = np.finfo(np.float64).eps
        # Optimization iteration counter
        self.iter_num = 0