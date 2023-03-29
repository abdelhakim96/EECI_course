# #################################################################
#
# Computaion of an invariant outer-epsilon approximation of the
# minimal Robust Positivly Invariant Set (mRPI),
# see Rakovic et al., Invariant Approximations of the Minimal Robust invariant Set. 
# IEEE Transactions on Automatic Control 50, 3 (2005), 406�410.
#
# #################################################################


from cvxopt import matrix, solvers
import numpy as np
import cdd
from qpsolvers import solve_qp

def InvariantApprox_mRPIset_lit(A, W, epsilon,A_w,b_w):
    # system dimension
    #n = W.Dim
    
    # initialization
    alpha = 0
    logicalVar = 1
    s = 0
    A= np.array(A)
    while logicalVar == 1:
        s = s + 1
        # alpha_0(s)
        # inequality representation of the set W: f_i'*w <= g_i , i=1,...,I_max
        
        f_i = np.array(A_w).T
        g_i = np.array(b_w)
        I_max = len(b_w)
        # call of the support function h_W
        h_W = np.zeros(I_max)
        for k in range(I_max):
            #print((A**s).transpose())
            #print(f_i[:, k])
            a = (A**s).transpose() @ f_i[:, k]
            
            h_W[k] = fkt_h_W(a, W, A_w, b_w)
        
        # output
        alpha_opt_s = max(h_W / g_i)
        alpha = alpha_opt_s
        
        # M(s)
        ej = np.eye(n)
        sum_vec_A = np.zeros(n)
        sum_vec_B = np.zeros(n)
        updt_A = np.zeros(n)
        updt_B = np.zeros(n)
        for k in range(s):
            for j in range(n):
                a = (A**(k)) @ ej[:, j]
                updt_A[j] = fkt_h_W(a, W, A_w, b_w)
                updt_B[j] = fkt_h_W(-a, W, A_w, b_w)
            sum_vec_A = sum_vec_A + updt_A
            sum_vec_B = sum_vec_B + updt_B
        Ms = max(np.maximum(sum_vec_A, sum_vec_B))
        
        # Interrupt criterion
        if alpha <= epsilon / (epsilon + Ms):
            logicalVar = 0
    
    # Fs
    Fs = Polyhedron('A', [], 'b', [], 'Ae', np.eye(n), 'be', np.zeros(n))
    for k in range(s):
        Fs = Fs + (A**k) * W
    
    # F_Inf approx
    F_alpha_s = 1 / (1 - alpha) * Fs
    
    return F_alpha_s, alpha, s

def fkt_h_W(a, W, A_w, b_w):
    # dimension of w
    W_a = np.array( W.get_generators()) #V-Representation
    W_v = W_a [:,1:]                   #Remove colomn of ones
    nn = len(W_v[1])
    
    
    
    # cost function
    c = np.array(-a)
    G = np.array(A_w)
    h = np.array(b_w)
    
    print(c)
    print(G)
    print(h)
    
    # optimization
    sol=solve_qp(np.array([]),c, G, h,np.array([]),np.array([]),np.array([]),np.array([]),solver='cvxopt')
    if sol['status'] == 'optimal':
        # output
        w_opt = np.array(sol['x']).reshape(nn,)
        h_W = a @ w_opt
    else:
        h_W = np.nan
    
    return h_W


