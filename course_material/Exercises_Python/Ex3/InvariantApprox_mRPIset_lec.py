#~*************************************************************************
# RPI_SET to determinate the RPI set for robust tube-based MPC 
# by fixing alpha and changing kappa. The algorithm has been explained in the lecture.
# S = RPI_Set(A_K,W,alpha,kappa)
# A_K: closed-loop A matrix of the system. it must be Schur stable
# W: disturbnic set 
# alpha: must be in the inteval [0,1)
#~*************************************************************************
##
import numpy as np
import cdd
from scipy.spatial import ConvexHull
import pypoman
import polytope as pc
from pytope import Polytope


#from ppl import Constraint, Constraint_System, Generator, Generator_System, Variable, C_Polyhedron, point, ray



def InvariantApprox_mRPIset_lec_solution(A_K, W, alpha, kappa):
    # Check alpha and kappa
    S_kappa =0 * W
    if (alpha <= 0) or (alpha >= 1):
       raise ValueError('alpha value must be in the interval (0,1)')
    elif kappa < 0:
       raise ValueError('kappa must be greater than or equal to zero')
    
    for i in range(kappa):
        M = np.linalg.matrix_power(A_K,kappa) * W   #Calculate S_kappa_array version
        S_kappa = S_kappa + M
    

    v1_C_v2=0 # initialize, A is not C of B
    
    #S_kappa =Polytope(np.zeros(len(W_v[0]),len(W_v[1])))
    while not (v1_C_v2 ):
          
          m_1 = (np.linalg.matrix_power(A_K,kappa) @ (np.array(W.V)).T).T 
          P_inter= pypoman.intersection.intersect_polygons(m_1, np.array(W.V)*alpha)
          M_inter=np.array(P_inter)

          # code to check intersection
          #check if p1 lies inside p2
          intersect = 1
        
          for j in range(len(m_1)):
              # get first vetrix in m1
              v_c = m_1[j,:]
              v_subset = np.around(v_c,6) in np.around(M_inter,6)
              if not(v_subset):
                 intersect = 0

            
          if intersect == 1:
              v1_C_v2 = 1
              
          
          
          M = np.linalg.matrix_power(A_K,kappa) * W
          
               
          S_kappa = S_kappa + M
          
          # Update kappa, S_kappa
          kappa = kappa + 1
          
   
    S = (1-alpha)**(-1)*S_kappa
 
    
    return S


