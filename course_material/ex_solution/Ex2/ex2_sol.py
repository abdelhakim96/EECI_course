import timeit

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as linalg

from run_ol import run_ol
from run_cl import run_cl
from run_cl_qih import run_cl_qih
from compute_alpha import computeAlpha_solution
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# for people who cannot see an interactive plot, uncomment the following lines
import matplotlib
if matplotlib.get_backend() == 'agg':
    matplotlib.use('WebAgg')
print(f'backend: {matplotlib.get_backend()}')

# MPC parameters
Q = np.diag([0.5,0.5])
R = np.diag([1.0])
u_max = 2
Tf = 2.0 # simulation horizon
T = 1.5  #prediction horizon
dt= 0.1  
x_init = np.vstack((-0.4,-0.5))
N = int(Tf/dt)
Ns = int(T/dt)
kappa = 1


# Problem 2
#Compute terminal region using both methods

kappa = 0.95
[K_loc,P,alpha] = computeAlpha_solution(kappa)


#Plot solution

run_ol(u_max, Q, R, Tf, x_init, N)   #open-loop with out terminal constraint

run_cl(u_max, Q, R, T, Tf, x_init, Ns)   #closed-loop with out terminal constraint

run_cl_qih(u_max, Q, R, T, Tf, x_init, Ns,alpha,P)   #quasi-infinite horizon MPC

plt.show()



##Plot Terminal Region

#A = np.array([[1,0],[0,2]])

El_1 = alpha * np.linalg.inv(P)
El_1 = np.array([[El_1[0][0],El_1[0][1],0],[El_1[1][0],El_1[1][1],0],[0,0,0]])


u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
zz = np.linspace(0, 1, 1)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones_like(u), np.ones_like(u))


El_c= 0.0
ellipsoid1 = (El_1 @ np.stack((x, y,z), 0).reshape(3, -1) + El_c).reshape(3, *x.shape)


fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')


ax.view_init(-120, -120)
ax.view_init(-90, 90)
ax.plot_surface(*ellipsoid1, rstride=4, cstride=4, color='b', alpha=0.1)

ax.scatter(0,0,0, marker= "*") # plot the point (2,3,4) on the figure

plt.show()

