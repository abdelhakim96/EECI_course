import matplotlib.pyplot as plt

# for people who cannot see an interactive plot, uncomment the following lines
import matplotlib
if matplotlib.get_backend() == 'agg':
    matplotlib.use('WebAgg')
print(f'backend: {matplotlib.get_backend()}')


# generate one common figure
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
#fig1, ax3 = plt.subplots(num=1)

ax1.grid(True)
ax1.set_title('MPC simulation')
ax1.set_ylabel(r'$x$')
ax2.grid(True)
ax2.set_xlabel(r'$t$')
ax2.set_ylabel(r'$u$')
line_colors = iter(['b', 'k', 'r'])


def plot_traj(t_grid, x_traj, u_traj, label):
    color = next(line_colors)
    ax1.plot(t_grid, x_traj, color=color, linestyle="--", linewidth=0.8)
    ax2.step(t_grid, u_traj, color=color, linestyle="--", linewidth=0.8, label=label)
    ax2.legend(loc="upper right")
    plt.draw()
    
    
    
    print(x_traj[:,0].T)
   # print(x_traj[:].T)
    
    plt.figure(0)
    plt.plot(x_traj[:,0].T, x_traj[:,1].T)

    #ax1.plot(x_traj[0], x_traj[1], color=color, linestyle="--", linewidth=0.8)
    #ax3.plot(x_traj[0], x_traj[1], color=color, linestyle="--", linewidth=0.8)
    #plt.plot(1)
    
    #ax1.plot(x_traj[:,0], x_traj, color=color, linestyle="--", linewidth=0.8)

