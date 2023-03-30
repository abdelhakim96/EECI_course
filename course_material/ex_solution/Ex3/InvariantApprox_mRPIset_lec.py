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






def InvariantApprox_mRPIset_lec_solution(A_K, W, alpha, kappa, W_pol):
    # Check alpha and kappa
    S_kappa =0 * W
    if (alpha <= 0) or (alpha >= 1):
       raise ValueError('alpha value must be in the interval (0,1)')
    elif kappa < 0:
       raise ValueError('kappa must be greater than or equal to zero')
    
    for i in range(kappa):
        M = np.linalg.matrix_power(A_K,kappa) * W   #Calculate S_kappa_array version
        S_kappa = S_kappa + M
    

    
    
    
    #Approach 1 using pc library to test if P1 C P2
    

    ap = (np.array(A_K)) * W


    W_alph= W*alpha
    A_alph= W_alph.A
    b_alph= W_alph.b
    W_poly_alpha= pc.Polytope(A_alph, b_alph)    
 
    exit = 0
    
    while not (exit):
          set_inc=0
          ap = np.linalg.matrix_power(A_K,kappa) * W
          
          for i in range(len(ap.V)):
              if not (ap.V[i,:]  in W_poly_alpha):
                  set_inc = 1
                  
          if set_inc == 0:
              exit = 1         
          kappa = kappa+1      
          S_kappa = S_kappa + np.linalg.matrix_power(A_K,kappa-1)*W
          
               
                  
    
      
    '''
    #Approach 2, also works
    v1_C_v2=0 # initialize, A is not C of B
    while not (v1_C_v2 ):
          
          m_1 = (np.linalg.matrix_power(A_K,kappa) @ (np.array(W.V)).T).T 
          # code to check intersection
          #check if p1 lies inside p2
          intersect = 1
        
          for j in range(len(m_1)):
              # get first vetrix in m1
              v_c = m_1[j,:]
              v_subset = np.around(v_c,8) in np.around(M_inter,8)
              if not(v_subset):
                 intersect = 0

            
          if intersect == 1:
              v1_C_v2 = 1
              
          
          
          M = np.linalg.matrix_power(A_K,kappa) * W
          M1 = np.linalg.matrix_power(A_K,kappa-1) * W
          kappa = kappa+1         
          S_kappa = S_kappa + M1
          
          # Update kappa, S_kappa
          
    '''
   
    S = (1-alpha)**(-1)*S_kappa
    print('kappa')
    print(kappa) 
    return S


