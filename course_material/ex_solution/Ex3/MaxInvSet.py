##
# computes the maximal output admissible set for a discrete time linear
# system based on [Gilbert, Tan: Linear Systems with State and Control
# Constraints: The Theory and Application of Maximal Output Admissible Sets,
# IEEE Transactions on Automatic Control, vol.36, No. 9, 1991


# autonomous system: x+ = Ax
# constraints:       H*x-h <= 0

# MOAS O_inf defined by G*x <= g



import numpy as np
from scipy.optimize import linprog
from cvxopt import matrix, solvers
from pytope import Polytope
import pypoman

def max_inv_set(A, H, h):
    """
    This function computes the maximal invariant set (MIS) for a discrete-time linear system with input constraints
    
    """
    m = len(h)
    
    notFinished = True
    fmax = -np.inf
    
    h_new = h.copy()
    H_new = H.copy()
    
    while notFinished:
        
        for i in range(m):
            
            
            c = np.array(-H_new[len(H_new)-m+i,:]@(A))
            res = linprog(c, A_ub=H_new, b_ub=h_new.T, bounds=(None, None))
            fval = res.fun
            
            fmax = max(-fval-h[i], fmax)
            print('fval')
            print(fval)
            
            print('fmax')
            print(fmax)
        
        if fmax <= 0:
            notFinished = False
        else:
            fmax = -np.inf
            H_new = np.vstack((H_new, H_new[len(H_new)-m+1,:]@(A)))
            h_new = np.hstack((h_new, h_new[len(h_new)-m+1]))
            
    G = H_new
    g = h_new
    
    O_Inf = np.array(pypoman.compute_polytope_vertices(G, g))
    
    
    return O_Inf, G, g
