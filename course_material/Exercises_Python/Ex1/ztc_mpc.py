import timeit

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np

from run_ol import run_ol
from run_cl import run_cl
from run_sqp import run_sqp


# for people who cannot see an interactive plot, uncomment the following lines
import matplotlib
if matplotlib.get_backend() == 'agg':
    matplotlib.use('WebAgg')
print(f'backend: {matplotlib.get_backend()}')


Q = np.diag([0.5,0.5])
R = np.diag([1.0])
u_max = 2
Tf = 5
x_init = np.vstack((0.4,-0.5))
#x_init = np.vstack((-0.7,-0.8))
N = 50
kappa = 2
# open loop
t_start = timeit.default_timer()
run_ol(u_max, Q, R, Tf, x_init, N)
t1 = timeit.default_timer() - t_start
# closed loop
t_start = timeit.default_timer()
run_cl(u_max, Q, R, Tf/kappa, Tf, x_init, N//kappa)
t2 = timeit.default_timer() - t_start
# SQP approximation
#t_start = timeit.default_timer()
#run_sqp(u_max, Q, R, Tf/kappa, Tf, x_init, N//kappa)
#t3 = timeit.default_timer() - t_start


print(
    f"\n"
    f"open loop time cost: {t1}s\n"
    f"closed loop time cost: {t2}s\n"
#    f"SQP approximation time cost: {t3}s\n"
)
plt.show()
