import timeit
import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
from run_ol import run_ol
from run_cl import run_cl
import matplotlib
if matplotlib.get_backend() == 'agg':
    matplotlib.use('WebAgg')
print(f'backend: {matplotlib.get_backend()}')


# Problem 1: open-loop MPC, with terminal constraint

# a) Define MPC parameters and initial condition
Q = np.diag([0.5,0.5])   #Weight matrices
R = np.diag([1.0])   
u_max = 2   # input limit 
Tf = 5      # prediction horizon
Ts = Tf     # simulation time
x_init = np.vstack((0.4,-0.5))   #initial condition
dt= 0.1  # sampling time
N = int(Tf/dt)    # number of steps 

t_start = timeit.default_timer()


run_ol(u_max, Q, R, Tf, x_init, N)   #solve ocp 



## Problem 2: closed-loop MPC with terminal constraint

run_cl(u_max, Q, R, Tf, Ts, x_init, N) #solve ocp 


# Problem 3 Try different initial states, prediction horizon lengths, weights Q and R , different
#constraint sets U , and add state constraints X , and analyze the changes.

plt.show()
