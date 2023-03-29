# Import the casadi library
from casadi import *

# Create an optimization object
opti = Opti()

# Define the system parameters
mu = 0.5
dynamics = lambda x,u: vertcat(x[1] + u*(mu - 4*(1 - mu)*x[2]), x[2] + u*(mu + (1 - mu)*x[1]))
n = 2
m = 1
delta = 0.1
T = 5
N = int(T/delta)
tmeasure = 0.0
x0 = MX([0.4, -0.5])
SimTime = 5
mpciterations = int(SimTime/delta)

# Define the cost function
Q = 0.5 * diag([1, 1])
R = 1

# Define the optimization variables
x = opti.variable(n, N+1)
u = opti.variable(m, N)

# Define the constraints
opti.subject_to(x[:, 0] == x0)
for k in range(N):
    k1 = dynamics(x[:, k], u[:, k])
    k2 = dynamics(x[:, k] + delta/2*k1, u[:, k])
    k3 = dynamics(x[:, k] + delta/2*k2, u[:, k])
    k4 = dynamics(x[:, k] + delta*k3, u[:, k])
    x_next = x[:, k] + delta/6*(k1 + 2*k2 + 2*k3 + k4)
    opti.subject_to(x[:, k+1] == x_next)
opti.subject_to(x[:, N+1] == MX.zeros(n))
opti.subject_to(u >= -2)
opti.subject_to(u <= 2)

# Define the cost function
J = 0
for i in range(N):
    J += x[:, i].T @ Q @ x[:, i] + u[:, i].T @ R @ u[:, i]
opti.minimize(J)

# Set initial guess for the optimization variables
u_init = np.zeros((m, N))
opti.set_initial(u, u_init)
x_init = np.zeros((n, N+1))
x_init[:, 0] = x0
for k in range(N):
    k1 = dynamics(x_init[:, k], 0)
    k2 = dynamics(x_init[:, k] + delta/2*k1, 0)
    k3 = dynamics(x_init[:, k] + delta/2*k2, 0)
    k4 = dynamics(x_init[:, k] + delta*k3, 0)
    x_init[:, k+1] = x_init[:, k] + delta/6*(k1 + 2*k2 + 2*k3 + k4)
opti.set_initial(x, x_init)

# Create a solver instance and solve the optimization problem
opti.solver('ipopt')
sol = opti.solve()

# Extract the optimal solution
x_OL = sol.value(x)
u_OL = sol.value(u)

# Plot the results
import matplotlib.pyplot as plt
tgrid = np.arange(0, SimTime+delta, delta)
plt.subplot(3, 1, 1)
plt.plot(tgrid, x_OL[0, :])
plt.xlabel('t')
plt.ylabel('x1(t)')
plt.xlim([0, T])
plt.subplot(3, 1, 2)
plt.plot(tgrid, x_OL[1, :])
plt.xlabel('t')
plt.ylabel('x2(t)')
