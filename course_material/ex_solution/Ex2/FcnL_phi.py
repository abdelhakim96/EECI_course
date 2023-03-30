import numpy as np
from scipy.optimize import minimize ,NonlinearConstraint, LinearConstraint




def FcnL_phi_func(AK, K, P, alpha):
    # Nonlinear system
    mu = 0.5 
 
    def lqr(x):
        return K@x

    def dynamics(x):
        mu = 0.5 
        return  np.array([x[1] + lqr(x) * (mu + (1-mu) * x[0]), x[0] + lqr(x) * (mu - 4*(1-mu)*x[1])])    
    

    phi = lambda x: [dynamics(x) - np.dot(AK,x).reshape(-1, 1)] 
    
    
    def nonlinConsAlpha(x):
        # All states inside ellipse X_{alpha}^f = {x \neq 0 | x'*P*x \leq alpha}
        c = x.T @ P @ x - alpha
        return c


    cons_func = lambda x: nonlinConsAlpha(x)
    obj_func = lambda x: - np.linalg.norm(phi(x)) / np.linalg.norm(x)

    

    res = minimize(obj_func, x0=[10, 10], method='SLSQP',constraints=[{'type': 'eq', 'fun': cons_func}], tol = 1e-6)

    L_Phi = -res.fun
    #print( res.x)
    
    return L_Phi



