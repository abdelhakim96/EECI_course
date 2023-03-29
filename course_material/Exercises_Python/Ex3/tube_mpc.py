import casadi as ca
import matplotlib.pyplot as plt
import numpy as np

from dynamics import dynamics
from plot_utils import plot_traj


def run_tube_mpc(u_max, Q, R, T, Tf, x_init, N,P, Ad,Bd,Hu,Hx,ku,kx,S,delta,K):
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



    d = np.zeros((2,N))
    # Formulate the NLP
    X0 = ca.MX.sym('X0', nx, 1)
    w.append(X0)
    w0.extend([0.1, 0.1])
    lbw.extend([-ca.inf, -ca.inf])
    ubw.extend([ca.inf,  ca.inf])

    g.append(X0 - x0_bar)
    lbg.extend([0, 0])
    ubg.append([0, 0])

    Xk = x0_bar
    for k in range(N):
        # New NLP variable for the control
        Uk = ca.MX.sym('U_'+str(k), nu, 1)
        w.append(Uk)
        w0.append(0.1)
        lbw.append(-u_max)
        ubw.append(u_max)

        # Integrate till the end of the interval
        #Xk_end =  F(Xk, Uk)
        #Xk_end = Ad*Xk +Bd*Uk
        Xk_end = Ad@Xk + Bd@Uk   # Linear system dynamics
        J += (Xk.T@Q@Xk + delta * R@Uk**2 + delta)
        
    
       

        # New NLP variable for state at end of interval
        Xk = ca.MX.sym('X_'+str(k+1), nx, 1)
        w.append(Xk)
        w0.extend([0.1, 0.1])
        lbw.extend([-ca.inf, -ca.inf])
        ubw.extend([ca.inf, ca.inf])


        
        # Add equality constraint  
        if k<N-1:
            g.append(Xk_end-Xk)
            lbg.extend([0,0])
            ubg.extend([0,0])
            
            g.append(Hx@Xk)
            lbg.extend([-ca.inf, -ca.inf,-ca.inf, -ca.inf])
            ubg.extend(kx.T)
            
            g.append(Hu@Uk)
            lbg.extend([-ca.inf, -ca.inf])
            ubg.extend(ku.T)

            

    J += Xk.T@P@Xk   #terminal cost


    # convert data format
    w = ca.vertcat(*w)
    w0 = ca.vertcat(*w0)
    lbw = ca.vertcat(*lbw)
    ubw = ca.vertcat(*ubw)
    g = ca.vertcat(*g)
    lbg = ca.vertcat(*lbg)
    ubg = ca.vertcat(*ubg)

    # Create an NLP solver with parameter p=x0_bar
    prob = {'f': J, 'x': w, 'g': g, 'p': x0_bar}
    solver = ca.nlpsol('solver', 'ipopt', prob)

    # Closed-loop solver with RTI for each step
    w_iter = np.zeros((w.rows(), N_sim))
    
    Z_cl = np.zeros((2, N_sim+1))
    Z_cl[:, 0] = x_init.flatten()
    X_cl = np.zeros((2, N_sim+1))
    X_cl[:, 0] = x_init.flatten()
    
    U_cl = np.zeros((1, N_sim+1))
    V_cl = np.zeros((1, N_sim+1))
    d = np.zeros((2, 1))
    
    for i in range(N_sim):
        sol = solver(x0=w0, lbg=lbg, ubg=ubg, lbx=lbw, ubx=ubw, p=Z_cl[:, i])
        sol = sol["x"].full().flatten()
          
        # disturbance
        d_min = -0.05
        d_max = 0.05
        #d[:,1] =  (d_max - d_min) * np.random.rand(2,1)
        
        w_iter[:, i] = sol
        Z_cl[:, i+1] = sol[3:5]
        V_cl[:, i] = sol[2] 
        
        
        U_cl[:, i] = V_cl[:, i] + K@(Z_cl[:,i] - Z_cl[:,i])
        X_cl[:, i+1] = Ad@Z_cl[:, i+1] + Bd@V_cl[:, i] 
        
    
        
    
    # plot
    t_grid = np.arange(N_sim+1)*T/N
    t_grid = np.arange(N_sim+1)*T/N
    
    
    #U_cl[:, i+1] = np.nan
    

    plot_traj(t_grid, X_cl.T, U_cl.T, 'tube-mpc')




