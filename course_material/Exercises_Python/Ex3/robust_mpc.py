##Code for implementation of Robust Tube based MPC from EECI course
##Mayne, D.Q., Seron, M.M., & Raković, S.V. (2005). Robust model predictive control of constrained linear systems \
# with bounded disturbances. Automatica, 41(2), 219–224.


import casadi as ca                # Import the casadi toolbox
import numpy as np 
from control import matlab
import control
import pypoman
import ppl 
import cdd
import polytope as pc
from pytope import Polytope
import matplotlib.pyplot as plt
#from tube_mpc import run_tube_mpc

from InvariantApprox_mRPIset_lec import InvariantApprox_mRPIset_lec_solution
from MaxInvSet import max_inv_set
from scipy.spatial import ConvexHull
from InvariantApprox_mRPIset_lit import InvariantApprox_mRPIset_lit

opti = ca.Opti()                   # Create a new optimization problem
opti.solver('ipopt')               # Choose a solver

## System dynamics

# Parameters
Rm = 2.7
L = 0.1
ku = 0.01
km = 0.05
J = 0.01
d = 0.1

# Sampling time
delta = 0.01

# continuous-time system                                        
Ac = np.array([[-Rm/L, -ku/L],
      [km/J,  -d/J]])
Bc = np.array([[1/L], [0]])


# Initial condition
x0 = [-2, 4] 

# Weighting matrices for cost function
Q = [[100, 0], [0, 1000]] 
R = [[1]]

# Prediction horizon
T = 0.05
N = int(T/delta)

## Problem 1: Exact discretization
n = Ac.shape[1] # state dimension
m = Bc.shape[1] # input dimension


# Exact discretization to obtain discrete-time system
# Ad = ???
# Bd = ???
#ss_d = control.ss(Ac,Bc,np.eye(n),np.zeros((n,m)),delta)  

#a= control.matlab.c2d(sysc, Ts, method='zoh', prewarp_frequency=None)
sysc= control.StateSpace(Ac, Bc, np.eye(n),np.zeros((n,m)))

sysd= control.matlab.c2d(sysc, delta, method='zoh', prewarp_frequency=None)

Ad = sysd.A
Bd = sysd.B


## Problem 2
# Derive K as the LQR controller and the weighting matrix P for the terminal cost function

# Discrete-time Algebraic Riccati Equation 
# P = ???
[P,L,G] = control.dare(Ad,Bd,Q,R)

# Discrete-time LQR controller
# HINT: We use u = Kx, whereas the Casadi function 'dlqr' uses u=-Kx
# K = ???
#K = -ca.dlqr(Ad,Bd,Q,R)
K, S, E = control.dlqr(sysd, Q, R)

K =-K

# LQR closed-loop dynamics 
# A_K = ???
A_K = Ad + Bd @ K


A_K =[[0.322024805356563,	-0.866180551029054],[0.0304586636399180, 0.882960946670225]]





A = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
b = np.array([3, 3, 10, 10])
# Create the inequality matrix
X =np.array( pypoman.compute_polytope_vertices(A, b))

X = Polytope(X)

#print(p.dim)

A=np.array([[1], [-1]])
b=np.array([10, 10])


#p2 = pc.Polytope(A, b)


U = np.array(pypoman.compute_polytope_vertices(A, b))
#U = np.hstack([U, U +1,U-1]).T


U = Polytope(U)

A=np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
b=np.array([0.05, 0.05, 0.05, 0.05])


W = np.array(pypoman.compute_polytope_vertices(A, b))
W = Polytope(W)
W_pol = pc.Polytope(A, b)
alpha = 0.25
kappa = 0
S = InvariantApprox_mRPIset_lec_solution(A_K,W,alpha,kappa,W_pol)

#get the set difference

print('S')
print(len(S.V))
Z=X-S


L=(np.matrix(K) * S)
 

A_l,b_l=pypoman.duality.compute_polytope_halfspaces(L.V)



p = pc.Polytope(A_l, b_l)
L =pc.extreme(p)           #minimal representation

V = np.array(U.V) + L   

V = Polytope(V)



Hx= Z.A
kx=Z.b



print('kx')
print(kx)
#Hu =V.A
#ku=V.b

Hx,kx = pypoman.duality.compute_polytope_halfspaces(Z.V)
Hu,ku = pypoman.duality.compute_polytope_halfspaces(V.V)


kx = np.array(kx)
ku= np.array(ku)

# Constraints to determine the terminal set {x|HXf*x<=KXf}


HXf = np.vstack((Hx, np.array(Hu*K)))
KXf = np.hstack([kx.T, np.array(ku).T])

# Terminal set Zf
# [Zf, Hf, Kf] = MaxInvSet(?,?,?)
[Zf, Hf, Kf] = max_inv_set(A_K, HXf, KXf)

Zf =Polytope(Zf)
print('hakim Hf!!')
print(Hf)
print(Kf)
'''
print('Z')
print(Z.V)

print('V')
print(V.V)

print('Matrices')
print(Ad)
print(Bd)
print(Hu)
print(Hx)
print(ku)
print(kx)
print(Hf)
print(Kf)
print(HXf)
print(KXf)print(Hf)

'''

######### Plotting 
# Figure 1


fig1, ax1 = plt.subplots(num=1)
plt.grid()
plt.axis([-3.5, 4.5, -11, 11])

X.plot(ax1, facecolor='red')
Zf.plot(ax1, facecolor='cyan')
S.plot(ax1, facecolor='yellow')
W.plot(ax1, facecolor='g')

plt.legend(['X', 'Zf', 'S', 'W'])
plt.show()




# Problem 4: Implementation of the MPC optimal control problem


Q = np.array([[100, 0],[0, 1000]])
R = np.diag([1.0])
u_max = 2
Tf = 0.15 # simulation horizon
T = 0.05  #prediction horizon
dt=0.01 
x_init = np.vstack((-2.0,4.0))
N = int(T/dt)
Ns = int(T/dt)
kappa = 1
#print(Ad)
#print(Bd)
#run_tube_mpc(u_max, Q, R, T, Tf, x_init, N, P, Ad, Bd, Hu, Hx, ku, kx, S,dt,K)


# Set number of MPC iterations
MPCIterations = 15
N= 5
# Define the optimization variables

z = opti.variable(n, N+1)
v = opti.variable(m, N)

# Dynamic constraint
for k in range(N-1):
    opti.subject_to(z[:, k+1] == Ad @ z[:, k] + Bd @ v[:, k])

# Input & state constraints
for k in range(N-1):
    opti.subject_to(Hu @ v[:, k] <= ku)
    opti.subject_to(Hx @ z[:, k+1] <= kx)



# Initial constraint
xt = opti.parameter(n, 1)
opti.set_value(xt, x0)
opti.subject_to(-S.A @ z[:, 0] <= S.b - S.A @ xt)




# Terminal constraint

opti.subject_to(Hf @ z[:, N] <= Kf)

# Cost function
J = 0
# Stage cost
for k in range(N-1):
    J +=  delta * (z[:,k].T) @ Q @ z[:,k] + delta * v[:,k].T@ R @ v[:,k] 
    
# Terminal cost
J += z[:, N].T @ P @ z[:, N]
opti.minimize(J)  # Minimize cost function


## Simulation of the closed loop
# Memory allocation

x_MPC = np.zeros((n, MPCIterations))
x_MPC[:,0] = x0
u_MPC = np.zeros((m, MPCIterations-1))
z_MPC = np.zeros((n, MPCIterations))
v_MPC = np.zeros((m, MPCIterations-1))

for ii in range(MPCIterations-1):
    # solve the optimization problem
    sol = opti.solve()
    # Extract the optimal nominal trajectories z*, v*
    # z_OL = ???
    # v_OL = ???
    # z_MPC = ???
    # v_MPC = ???
    z_OL = sol.value(z)
    v_OL = sol.value(v)
    #print(v_OL)
    z_MPC[:,ii] = z_OL[:,0]
    v_MPC[0,ii] = v_OL[0]
  
    # Application of the tube-based MPC control law
    u_MPC[:,ii] = v_MPC[:,ii] + K @ (x_MPC[:,ii] - z_MPC[:,ii])
    
    # Disturbance at this iteration (random point in W)
    w_min = -0.05
    w_max = 0.05
    w = np.random.uniform(w_min, w_max, n)
    #w = w_min + (w_max - w_min) * np.random.rand(1,2)
    print('w')
    print(w)
    # Update the closed-loop system
    x_MPC[:,ii+1] = Ad @ x_MPC[:,ii] + Bd @ u_MPC[:,ii] + w
    
    
   
    # Prepare warmstart solution for next time step (take the endpiece of the optimal solution 
    # and add a last piece) 
    # z_init = ???
    # v_init = ???
   # z_init = np.hstack((z_OL[:,1:], A_K @ z_OL[:,-1][:,np.newaxis]))
   
    a = (A_K @ z_OL[:,len(z_OL)])
    a = a.reshape((1,-1)).T
    z_init = np.hstack((z_OL[:,1:], a))
    
    #print((v_OL[1:]))
   # print(K @ z_OL[:,-1][:,np.newaxis])
    
    #v_init = np.vstack((v_OL[1:], K @ z_OL[:,-1][:,np.newaxis]))
    
   # print(K @ z_OL[:,-1][:,np.newaxis])
    aa = np.array((K @ z_OL[:,len(z_OL)])).flatten()
    #aa = aa.reshape((1,-1)).T 
    print('vstack')
    print(v_OL[1:])
    print(aa)
    v_init = np.hstack((v_OL[1:], aa))
    opti.set_initial(z, z_init)
    opti.set_initial(v, v_init)

   
    # update initial constraint parameter
    opti.set_value(xt,x_MPC[:,ii+1])
    #xt.value = x_MPC[:,ii+1]
    
# Plots
# Plot the states of the real (disturbed) system and the sequence of the
# initial states of the nominal system
# Plot terminal Set
print('z_OL')
print(z_OL)
#print(z_MPC)
S_a = np.array(S.V)

fig = plt.figure(3)
ax = fig.add_subplot(111)
S.plot(ax=ax)

ZS =Zf+S

print('U')
print(U.V)


ZS.plot(ax, color='cyan')
#ax.plot(Zf+S, color='cyan'), 
Zf.plot(ax, color='red')
for ii in range(MPCIterations-2):
    safe_v= S.V+z_MPC[:,ii]
    #A,b = pypoman.duality.compute_polytope_halfspaces(safe)
    Safe_P = Polytope(np.array(safe_v))
    Safe_P.plot(ax, facecolor='yellow', edgecolor='k', edgealpha=0.1)



ax.plot(x_MPC[0,:], x_MPC[1,:])
ax.plot(z_MPC[0,:], z_MPC[1,:], 'g-.')
ax.plot(0,0,'x',color='black')
ax.set_title('state space')
ax.set_xlabel('x_1')
ax.set_ylabel('x_2')
plt.grid()
plt.show()



