import casadi as ca
import matplotlib.pyplot as plt
import numpy as np

from dynamics import dynamics
from plot_utils import plot_traj


def run_sqp(u_max, Q, R, T, Tf, x_init, N):
    # N = 50 # number of control intervals
    nx = 2
    nu = 1
    N_sim = round(Tf/(T/N))

    # Declare model variables
    x = ca.MX.sym('x',nx,1)
    u = ca.MX.sym('u',nu,1)

    x0_bar = ca.MX.sym('x0_bar',nx,1)

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

    # Formulate the NLP
    X0 = ca.MX.sym('X0', nx, 1)
    Xk = X0
    w.append(Xk)
    w0.append(x0_bar)
    lbw.extend([-ca.inf, -ca.inf])
    ubw.extend([ca.inf,  ca.inf])

    g.append(Xk - x0_bar)
    lbg.extend([0, 0])
    ubg.append([0, 0])

    for k in range(N):
        # New NLP variable for the control
        Uk = ca.MX.sym('U_'+str(k), nu, 1)
        w.append(Uk)
        # w0.append(0.1)
        lbw.append(-u_max)
        ubw.append(u_max)

        # Integrate till the end of the interval
        Xk_end =  F(Xk, Uk)
        # J +=1/2*(Xk.T@Q@Xk + R@Uk**2)

        # New NLP variable for state at end of interval
        Xk = ca.MX.sym('X_'+str(k+1), nx, 1)
        w.append(Xk)
        # w0.extend([0.1, 0.1])
        lbw.extend([-ca.inf, -ca.inf])
        ubw.extend([ca.inf, ca.inf])

        # Add equality constraint
        g.append(Xk_end-Xk)
        lbg.extend([0,0])
        ubg.extend([0,0])
    # J += 1/2*Xk.T@Q@Xk

    # convert data format
    w = ca.vertcat(*w)
    w0 = ca.vertcat(*w0)
    lbw = ca.vertcat(*lbw)
    ubw = ca.vertcat(*ubw)
    g = ca.vertcat(*g)
    lbg = ca.vertcat(*lbg)
    ubg = ca.vertcat(*ubg)

    # utility function
    G = ca.Function('G', [w, x0_bar], [g])
    JG = ca.Function('JG',  [w, x0_bar], [ca.jacobian(g,w)])

    # linearize the constraints
    wk = ca.MX.sym('wk', w.rows(), 1)
    g_l = G(wk, x0_bar) + JG(wk, x0_bar)@(w-wk)

    # hessian matrix
    H = np.kron(np.eye(N), np.diag([*np.diag(Q), *np.diag(R)]))
    H = np.diag([*np.diag(H), *np.diag(Q)])
    J = 1/2*w.T@H@w

    # allocate QP solver
    prob = {'f': J, 'x': w, 'g': g_l, 'p': ca.vertcat(wk, x0_bar)}
    solver = ca.qpsol('solver', 'qpoases', prob)

    # Closed-loop solver with RTI for each step
    w_iter = np.zeros((w.rows(), N_sim))
    X_cl = np.zeros((2, N_sim+1))
    X_cl[:, 0] = x_init.flatten()
    U_cl = np.zeros((1, N_sim+1))

    for i in range(N_sim):
        sol = solver(lbg=lbg, ubg=ubg, lbx=lbw, ubx=ubw, p=np.append(w_iter[:,i], X_cl[:, i]))
        sol = sol["x"].full().flatten()
        w_iter[:, i] = sol
        U_cl[:, i] = sol[2]
        X_cl[:, i+1] = F(X_cl[:,i], U_cl[:,i]).full().flatten()
    
    # plot
    t_grid = np.arange(N_sim+1)*T/N
    U_cl[:, i+1] = np.nan
    plot_traj(t_grid, X_cl.T, U_cl.T, 'SQP')


# for test one single run
if __name__  == '__main__':
    Q = np.diag([10,0.1])
    R = np.diag([0.1])
    u_max = 5
    Tf = 3
    x_init = np.vstack((1,3))
    N = 20
    kappa = 4

    run_sqp(u_max, Q, R, Tf/kappa, Tf, x_init, N//kappa)
    plt.show()
