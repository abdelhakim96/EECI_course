import casadi as ca
import numpy as np
import control
#from qpsolvers import solve_qp
from cvxopt import matrix,solvers
from scipy.optimize import minimize

#def computeAlpha_solution(kappa):

from FcnL_phi import FcnL_phi_func

#kappa=-20
def computeAlpha_solution(kappa):



    mu= 0.5
    def lqr(x):
        return K@x

    def dynamics(x):
        mu = 0.5 
        return  np.array([x[1] + lqr(x) * (mu + (1-mu) * x[0]), x[0] + lqr(x) * (mu - 4*(1-mu)*x[1])])    

    u_max=2
    u_min=-2


    print('Step 1')
    #Compute linearized system


    A = np.array([[0, 1], [1, 0]])
    B = np.array([[0.5], [0.5]])


    n=2
    m=1
    # LQR controller
    Q = 0.5 * np.eye(2)
    R = np.eye(1)

    # compute K
    K, S, E =  control.lqr(A, B, Q, R)
    
    K = -K 

    AK = A + B @ K

    print(f"K = {np.array2string(K, separator=', ')}")
    
    # Compute the eigenvalues of AK
    eigvals, _ = np.linalg.eig(AK)

    # Display the eigenvalues
    print("eig(AK) = ")
    print(eigvals)
    print("______________________________")

    print('Step 2')

    if kappa >= -max(np.real(np.linalg.eig(AK)[0])):
        raise ValueError('kappa >= -max(real(eig(AK)))')


    P = control.lyap(AK+kappa*np.eye(n), Q + K.T @ R @ K)
    #P= [P]
    print('P = ')
    print(P)
    print('______________________________')
    print('Step 3')
    

    u_max = 2* np.eye(1)    
    u_min = -u_max # example lower bound



    # Define the quadratic programming problem
    p = matrix([0.0, 0.0])
    A = matrix(K)
    G = matrix([[0.0,0.0],[0.0,0.0]])    #G = None
    h=matrix([0.0, 0.0])
    b1 = matrix(u_max)
    b2 = matrix(u_min)
    P = matrix(P)

    #sol1 = solvers.qp(Q=P, p=p, G=G,h=h, A=A, b1=b1)
    sol1=solvers.qp(P, p, G, h, A, b1)
    sol2=solvers.qp(P, p, G, h, -A, -b2)

    # Compute the minimum value of the objective function
    x_opt_1 = matrix(sol1['x'])
    x_opt_2 = matrix(sol2['x'])
    
    a1=x_opt_1.T * P * x_opt_1
    a2= x_opt_2.T * P * x_opt_2

    alpha_1 = min(a1[0], a2[0])
 

    print('x_opt_1')
    print(x_opt_1) 
    

    print('alpha_1')
    print(alpha_1) 



    print('______________________________')
  
    print('Step 4')

    # Step 4: Find largest possible alpha in (0,alpha_1] such that (3) holds
    # In this step, we search for a alpha<=alpha_l, such that
    # L_Phi<=L_Phi_max.
    # This is done using bisection.


    # Definition of phi in (1)

    phi = lambda x: [dynamics(x) - np.dot(AK,x).reshape(-1, 1)] 

   

    # upper bound for L_Phi
    L_Phi_max = (kappa * np.min(np.real(np.real(np.linalg.eig(P)[0])))) / (np.linalg.norm(P, ord=2))


    # initial conditions for optimization
    alpha_ub = alpha_1  # upper bound for alpha
    alpha_lb = 0  # lower bound for alpha
    L_Phi = FcnL_phi_func(AK, K, P, alpha_1)  # Compute L_phi for alpha_1   
    exitflag = 1
    nn = 1
    n_max = 100  # maximum number of iterations

   # print(L_Phi)
    # Check upper bound for alpha_1
    if (L_Phi <= L_Phi_max):
        alpha_lb = alpha_1
        exitflag = 0  # End bisection if alpha_1 is small enough
    alpha = 0.5 * alpha_1  # Next guess for alpha
    L_Phi = FcnL_phi_func(AK, K, P, alpha)  # Compute L_phi for next guess

   
    #bisection
    while exitflag == 1 and nn <= n_max:
        alpha_old = alpha
        
        if L_Phi > L_Phi_max:                  # alpha is too big
            alpha_ub = alpha                   # new upper bound
        elif L_Phi <= L_Phi_max and L_Phi != 0: # alpha too small
            alpha_lb = alpha                   # new lower bound
        else:
            raise ValueError('error')
        
        alpha = 0.5*(alpha_ub + alpha_lb)      # New guess by bisection
        L_Phi = FcnL_phi_func(AK,K,P,alpha)         # Compute L_phi for new guess
        
        # exit conditions
        if abs(alpha - alpha_old)/abs(alpha_old) <= 10**-12 and L_Phi <= L_Phi_max and L_Phi != 0:
            exitflag = 0
        nn += 1
        
    alpha = alpha_lb                           # alpha = lower bound that satisfies < L_phi_max
    print('alpha new = ',alpha)
    


     

    
    ## Step 4 (alternative)
    alternative_procedure = True

    def nonlcon(x):
        # Inequality constraint x'Px <= alpha
        c = -(x.T @ P @ x - alpha)
        #c=0
        
        return c


   # if alternative_procedure:

    print('Step 4 (alternative)')
    #  Set lower and upper bounds for alpha for bisection
    alpha = alpha_1
    print( 'alpha_init')
    print( alpha)
    alpha_ub = alpha_1
    alpha_lb = 0
    max_iter = 10                      # Max iterations for bisection
    opt = {'maxfun':10000, 'maxiter':10000, 'disp':False} # Options for fmincon
    x2_init = np.array([[1, 1, -1, -1], [1, -1, 1, -1]])   # Initialization for slsqp



    obj_func = lambda x: -(x.T @ P @ np.array(phi(x)).reshape(-1, 1) - kappa * x.T @ P @ x) / (x.T @ P @ x)


    #start bisection
    for i in range(max_iter):
        fval = np.inf
        # inner for loop to check multiple initializations for minimize
        for j in range(4):
    
            res = minimize(obj_func, x2_init[:,j], method='SLSQP',constraints=[{'type': 'ineq', 'fun': nonlcon}], tol = 1e-12)
            x2 = res.x
            val = res.fun 
            # Take minimum value
            fval = min(fval, val)
        fval =-fval  # Correct sign due to maximization    
        # Check condition: Is optimal value of (7) nonpositive and update upper and lower bounds for alpha  
        if fval >= 0: # condition not satisfied
            alpha_ub = alpha
        else: # condition satisfied
            alpha_lb = alpha
   

        alpha = (alpha_lb + alpha_ub) / 2  # bisection
    alpha = alpha_lb  # Take best known feasible value for alpha
    print('alpha_alt')
    print(alpha)
    print(alpha_ub)
    print(alpha_lb)
    
    return K,np.array(P),alpha    



    
 












