import timeit

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as linalg

from run_ol import run_ol
from run_cl import run_cl
from run_cl_qih import run_cl_qih
#from run_cl import run_cl
from run_sqp import run_sqp
from compute_alpha import computeAlpha_solution
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# for people who cannot see an interactive plot, uncomment the following lines
import matplotlib
if matplotlib.get_backend() == 'agg':
    matplotlib.use('WebAgg')
print(f'backend: {matplotlib.get_backend()}')

# problem 1 no terminal constraint
Q = np.diag([0.5,0.5])
R = np.diag([1.0])
u_max = 2
Tf = 2.0 # simulation horizon
T = 1.5  #prediction horizon
dt=0.1  
x_init = np.vstack((-0.4,-0.5))
N = int(Tf/dt)
Ns = int(T/dt)
kappa = 1

# open loop
#t_start = timeit.default_timer()
#run_ol(u_max, Q, R, Tf, x_init, N)
#t1 = timeit.default_timer() - t_start


# closed loop
#t_start = timeit.default_timer()
#run_cl(u_max, Q, R, T, Tf, x_init, Ns)
#plt.show()
#t2 = timeit.default_timer() - t_start


# SQP approximation
#t_start = timeit.default_timer()
#run_sqp(u_max, Q, R, Tf/kappa, Tf, x_init, N//kappa)
#t3 = timeit.default_timer() - t_start


#print(
  #  f"\n"
   # f"open loop time cost: {t1}s\n"
    #f"closed loop time cost: {t2}s\n"
   # f"SQP approximation time cost: {t3}s\n"
#)





# Problem 2
#Terminal set and cost
kappa = 0.95
[K_loc,P,alpha] = computeAlpha_solution(kappa)
#K_loc = np.array([-2.118, -2.118])
#print ('P')
#print(P)


#P1 = np.array([[16.5926, 11.5926], [11.5926, 16.5926]])
#alpha = 0.2495
#print ('P1')
#print(P1)



#problem 3 re-implement MPC with terminal constraints (QIH_MPC)


run_cl_qih(u_max, Q, R, T, Tf, x_init, Ns,alpha,P)
run_cl(u_max, Q, R, T, Tf, x_init, Ns)

#t2 = timeit.default_timer() - t_start
#run_cl(u_max, Q, R, T, Tf, x_init, Ns)
plt.show()





'''
##Plot Ellipsoid

#A = np.array([[1,0],[0,2]])

El_1 = alpha * np.linalg.inv(P)
El_1 = np.array([[El_1[0][0],El_1[0][1],0],[El_1[1][0],El_1[1][1],0],[0,0,0]])

#El_2 = 0.5* alpha * np.linalg.inv(P)
#El_2 = np.array([[El_2[0][0],El_2[0][1],0],[El_2[1][0],El_2[1][1],0],[0,0,0]])


#A = np.array([[1,0,0],[0,2,0],[0,0,1]])

u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
zz = np.linspace(0, 0.000001, 100)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones_like(u), np.ones_like(u))


El_c= 0.0
ellipsoid1 = (El_1 @ np.stack((x, y,z), 0).reshape(3, -1) + El_c).reshape(3, *x.shape)
#ellipsoid2 = (El_2 @ np.stack((x, y,z), 0).reshape(3, -1) + El_c).reshape(3, *x.shape)

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')
#ax.view_init(elev = 5, azim=180)

ax.view_init(-120, -120)
ax.view_init(-90, 90)
ax.plot_surface(*ellipsoid1, rstride=4, cstride=4, color='b', alpha=0.1)

ax.scatter(0,0,0, marker= "*") # plot the point (2,3,4) on the figure
#ax.plot_surface(*ellipsoid2, rstride=4, cstride=4, color='r', alpha=0.1)
plt.show()
'''