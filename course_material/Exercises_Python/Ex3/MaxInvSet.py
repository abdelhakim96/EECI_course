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
    using the linear programming approach.
    
    Parameters:
    A: numpy array of size n x n
       The state transition matrix of the system.
       
    H: numpy array of size m x n
       The input constraint matrix of the system.
       
    h: numpy array of size m x 1`
       The input constraint vector of the system.
       
    Returns:
    O_Inf: Polyhedron object
           The maximal invariant set computed using the given system dynamics and input constraints.
           
    G: numpy array of size p x n
       The matrix defining the polyhedron constraints in the half-space representation.
       
    g: numpy array of size p x 1
       The vector defining the polyhedron constraints in the half-space representation.
    """
    m = len(h)
    
    notFinished = True
    fmax = -np.inf
    
    h_new = h.copy()
    H_new = H.copy()
    
    while notFinished:
        
        for i in range(m):
            c = -H_new[-m+i,:].dot(A)
            #print(c)
           # print(h_new)
           # print(len(H_new[0]))
           # print(len(h_new))
            res = linprog(c, A_eq=H_new, b_eq=h_new.T, bounds=(None, None))
            fval = res.fun
            fmax = max(-fval-h[i], fmax)
        
        if fmax <= 0:
            notFinished = False
        else:
            fmax = -np.inf
            H_new = np.vstack((H_new, H_new[-m:,:].dot(A)))
            h_new = np.vstack((h_new, h_new[-m:,:]))
            
    G = H_new
    g = h_new
    
    O_Inf = np.array(pypoman.compute_polytope_vertices(G, g))

    
    return O_Inf, G, g
