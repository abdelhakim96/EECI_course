import numpy as np
import matplotlib.pyplot as plt
from casadi import *

opti = casadi.Opti()  # Create a new optimization problem
opti.solver('ipopt')  # Choose a solver

# System parameters
mu = 0.5
dynamics = lambda x, u: [x[1] + u * (mu + (1 - mu) * x[0]), x[0] + u * (mu - 4 * (1 - mu) * x[1])]
n = 2  # State dimension
m = 1  # Input dimension
delta = 0.1  # Sampling time
T = 1.5  # continuous-time prediction horizon
N = int(T / delta)  # discrete-time prediction horizon
tmeasure = 0.0  # initial time
x0 = [-0.4, -0.5]  # initial state
SimTime = 2  # Simulation time
mpciterations = int(SimTime / delta)  # discrete simulation time
Q = 0.5 * np.eye(n)
R = 1

# Problem 1
# Define optimization variables
x = opti.variable(n, N + 1)
u = opti.variable(m, N)
# Constraints
# dynamic constraints
for k in range(N):
    opti.subject_to(x[:, k + 1] == vertcat(*dynamics(x[:, k], u[:, k])))
# Initial constraint
opti.subject_to(x[:, 0] == x0)
# input constraint
opti.subject_to(opti.bounded(-1, u[0, :], 1))
# cost function
J = 0
for k in range(N):
    J += mtimes([x[:, k].T, Q, x[:, k]]) + mtimes([u[:, k].T, R, u[:, k]])
J += mtimes([x[:, N].T, P, x[:, N]])
opti.minimize(J)
# Initial guess
opti.set_initial(x, np.tile(np.array(x0).reshape((-1, 1)), (1, N + 1)))
opti.set_initial(u, np.zeros((m, N)))
# Storage
x_MPC = np.zeros((n, mpciterations + 1))
x_MPC[:, 0] = x0
u_MPC = np.zeros((m, mpciterations))
t = 0
# closed-loop iterations
f2 = plt.figure(2)
plt.plot(0, 0, 'rx')
for ii in range(mpciterations):
    sol = opti.solve()
    x_OL = sol.value(x)
    u_OL = sol.value(u)
    # store closed-loop data
    u_MPC[ii] = u_OL[:, 0]
    x_MPC[:, ii + 1] = x_OL[:, 1]
    t += delta
    # update initial constraint
    opti.set_initial(x, np.hstack((x_OL[:, 1:], x_OL[:, -1].reshape((-1, 1)))))
    # update initial guess
    opti.set_initial(u, np.hstack((u_OL[:, 1:], u_OL[:, -1].reshape((-1, 1)))))
    # Plot state sequences (open-loop and closed-loop) in state space plot (x_1;x_2)
    plt.figure(2)
   
