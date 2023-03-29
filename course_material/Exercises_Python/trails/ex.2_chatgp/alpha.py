import numpy as np
from scipy.optimize import minimize


def nonlinConsAlpha(x, P, alpha):
    """All states inside ellipse X_{\alpha}^f = {x \neq 0 | x'*P*x \leq alpha}"""

    c = np.dot(x.T, np.dot(P, x)) - alpha
    ceq = []

    return c, ceq


def FcnL_phi(AK, K, P, alpha):
    """Nonlinear system"""

    mu = 0.5
    dynamics = lambda x, u: np.array([x[1] + u * (mu + (1 - mu) * x[0]),
                                      x[0] + u * (mu - 4 * (1 - mu) * x[1])])

    # Phi
    def phi(x):
        return dynamics(x, K @ x) - AK @ x

    # L_Phi
    opt = {'maxiter': 10000, 'maxfev': 10000, 'disp': False}
    res = minimize(lambda x: -np.sqrt(np.dot(phi(x).T, phi(x))) / np.sqrt(np.dot(x.T, x)),
                   np.array([10, 10]),
                   constraints=[{'type': 'ineq', 'fun': lambda x: nonlinConsAlpha(x, P, alpha)[0]}],
                   options=opt)

    L_Phi = -res.fun

    return L_Phi
