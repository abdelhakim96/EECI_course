import casadi as ca
import matplotlib.pyplot as plt
import numpy as np

from dynamics import dynamics
from plot_utils import plot_traj


def run_ol(u_max, Q, R, T, x_init, N):
    # N = 50 # number of control intervals
    nx = 2
    nu = 1

    # Declare model variables
    x = ca.MX.sym('x',nx,1)
    u = ca.MX.sym('u',nu,1)

    x0_bar = x_init

    # Model equations
    xdot = dynamics(x, u)

    # Formulate discrete time dynamics
    # Fixed step Runge-Kutta 4 integrator
    M = 1 # RK4 steps per interval
    DT = T/N/M
    f = ca.Function('f', [x, u], [xdot])
    X0 = ca.MX.sym('X0',nx,1)
    U = ca.MX.sym('U',nu,1)
    X = X0

    for _ in range(M):
        k1 = f(X, U)
        k2 = f(X + DT/2 * k1, U)
        k3 = f(X + DT/2 * k2, U)
        k4 = f(X + DT * k3, U)
        X += DT/6*(k1 +2*k2 +2*k3 +k4)

    F = ca.Function('F', [X0, U], [X])

    # Evaluate at a test point
    Fk = F(np.vstack((0.2, 0.3)), 0.4)
    print(Fk)

    # Start with an empty NLP
    w=[]
    w0 = []
    lbw = []
    ubw = []
    J = 0
    g=[]
    lbg = []
    ubg = []

    X0 = ca.MX.sym('X0', nx, 1)
    w.append(X0)
    w0.extend([0.1, 0.1])
    lbw.extend([-ca.inf, -ca.inf])
    ubw.extend([ca.inf,  ca.inf])

    g.append(X0)
    lbg.append(x0_bar)
    ubg.append(x0_bar)

    # Formulate the NLP
    Xk = x0_bar
    for k in range(N):
        # New NLP variable for the control
        Uk = ca.MX.sym('U_'+str(k), nu, 1)
        w.append(Uk)
        w0.append(0.1)
        lbw.append(-u_max)
        ubw.append(u_max)

        # Integrate till the end of the interval
        Xk_end =  F(Xk, Uk)
        J += (Xk.T@Q@Xk + R@Uk**2)

        # New NLP variable for state at end of interval   (terminal constraint)
        Xk = ca.MX.sym('X_'+str(k+1), nx, 1)
        w.append(Xk)
        w0.extend([0.1, 0.1])
        lbw.extend([-ca.inf, -ca.inf])
        ubw.extend([ca.inf, ca.inf])

        # Add equality constraint  
        if k<N+1:
            g.append(Xk_end-Xk)
            lbg.extend([0,0])
            ubg.extend([0,0])
        #else:
        #    g.append(Xk_end)
        #    lbg.extend([0,0])
        #    ubg.extend([0,0])
        #    g.append(Xk)
        #    lbg.extend([0,0])
        #    ubg.extend([0,0])
            

    J += Xk.T@Q@Xk

    # convert data format
    w = ca.vertcat(*w)
    w0 = ca.vertcat(*w0)
    lbw = ca.vertcat(*lbw)
    ubw = ca.vertcat(*ubw)
    g = ca.vertcat(*g)
    lbg = ca.vertcat(*lbg)
    ubg = ca.vertcat(*ubg)

    # Create an NLP solver
    prob = {'f': J, 'x': w, 'g': g}
    solver = ca.nlpsol('solver', 'ipopt', prob)

    # Solve the NLP
    sol = solver(x0=w0, lbg=lbg, ubg=ubg, lbx=lbw, ubx=ubw)

    w_opt = sol["x"].full()
    x1_opt = w_opt[0:None:3]
    x2_opt = w_opt[1:None:3]
    u_opt = w_opt[2:None:3]

    # Plot the solution
    t_grid = np.linspace(0, T, N+1)
    x_grid = np.column_stack((x1_opt, x2_opt))
    u_grid = np.append(u_opt, np.nan)
    plot_traj(t_grid, x_grid, u_grid, label='open-loop (no TC)')


# for test one single run
if __name__ == "__main__":
    Q = np.diag([10,0.1])
    R = np.diag([0.1])
    u_max = 5
    Tf = 3
    x_init = np.vstack((1,3))
    N = 20
    kappa = 4

    run_ol(u_max, Q, R, Tf, x_init, N)
    plt.show()
