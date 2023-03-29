import casadi as ca
import matplotlib.pyplot as plt
import numpy as np

from rk4step import rk4step
from animate import animate_sys

# for people who cannot see an interactive plot, uncomment the following lines
import matplotlib
if matplotlib.get_backend() == 'agg':
    matplotlib.use('WebAgg')
print(f'backend: {matplotlib.get_backend()}')


# parameter for dynamics
mu= 0.1


## parameters required for the OCP
nx = 2  # state dimension
nu = 1  # control dimension
T= 1    # prediction horizon
dt = 0.1  # discretization time step
N = int(T/dt)  # number of steps
N_rk4 = 10  # number of rk4 steps per discretization time step


#x0_bar = np.array([0.4, -0.5])   # initial state

x0_bar = ca.MX.sym('x0_bar',nx,1)

## cost function params
Q = 0.5 * np.eye(nx)
R = 1.0


## constraints
u_max = 2.0

## build the integrator
# ode
def dynamics(x, u):
    return ca.vertcat(x[1]+ u*(mu +(1-mu)*x[0]), x[0] + u*(mu -4 * (1 - mu) *x[1]))

# discrete dynamics
x = ca.SX.sym('x', nx, 1)
u = ca.SX.sym('u', nu, 1)
x_next = x
for _ in range(N_rk4):
    x_next = rk4step(x, u, dynamics, dt)

# integrator
F = ca.Function('F', [x, u], [x_next])


## NLP formulation
x_traj = ca.SX.sym('x_traj', nx*N, 1)  # vector of all states (1,..,N)
u_traj = ca.SX.sym('u_traj', nu*N, 1)  # vector of all controls (0,..,N-1)

x_traj_init = .1 * np.ones((nx*N, 1))  # intial guess
u_traj_init = .1 * np.ones((nu*N, 1))  # intial guess


# some utils
u_traj_per_stage = ca.vertsplit(u_traj, nu)
x_traj_per_stage = ca.vertsplit(x_traj, nx)
stage_cost = ca.Function('stage_cost', [x, u], [x.T @ x + R * u.T @ u])
terminal_cost = ca.Function('terminal_cost', [x], [ x.T @ x])

# build cost and constraints
x_current = x0_bar
constraints = []
cost = 0
for i in range(N):
    u_current = u_traj_per_stage[i]
    x_next = x_traj_per_stage[i]

    constraints.append(x_next - F(x_current, u_current))
    cost += stage_cost(x_current, u_current)

    x_current = x_next
# terminal cost
cost += terminal_cost(x_current)


constraints = ca.vertcat(*constraints)

lbg.append(x0_bar)
ubg.append(x0_bar)

## nlp solver
nlp_sim = {'x': ca.vertcat(x_traj, u_traj), 'f': cost, 'g':constraints, 'p': x0_bar}
solver_sim = ca.nlpsol('solver','ipopt', nlp_sim)


# solve nlp
max_xu = np.vstack((
    np.inf*np.ones(x_traj.shape),  # no constraint on x
    u_max*np.ones(u_traj.shape)          # constraint on u
))
sol_sim = solver_sim(
    x0=ca.vertcat(x_traj_init, u_traj_init),
    lbx=-max_xu,
    ubx= max_xu,
    lbg=np.zeros(constraints.shape),
    ubg=np.zeros(constraints.shape),
)


## visualize solution
x_u_traj_opt_sim = sol_sim['x'].full().T
x_traj_opt_sim = ca.reshape(
    ca.vertcat(x0_bar, x_u_traj_opt_sim[:, 0:N*nx].T),
    (nx, N+1)
)
u_traj_opt_sim = x_u_traj_opt_sim[:, N*nx:None]

plt.figure(2)
plt.subplot(2,1,1)
plt.plot(x_traj_opt_sim.T)
plt.title('state trajectory')
plt.legend([r'$x1$', r'$x2$'])

plt.subplot(2,1,2)
plt.step(range(N), u_traj_opt_sim.T, where='post')
plt.title('control trajectory')
plt.legend([r'$u$'])
plt.xlabel('discrete time $k$')


ani = animate_sys(x_traj_opt_sim.full()[0,:] ,x_traj_opt_sim.full()[1,:])


plt.show()
