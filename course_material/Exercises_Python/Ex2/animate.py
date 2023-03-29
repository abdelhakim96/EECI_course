import numpy as np
import matplotlib.pylab as plt
import matplotlib.animation as animation

def animate_sys(x1, x2, dt=0.001):
    '''
    Create animation of a pendulum, where Theta contains the trajectory of its
    angle. dt defines the time gap (in seconds) between two succesive entries.

    Theta should be a list or 1D-numpy array
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, autoscale_on=False, xlim=(-1.2, 1.2), ylim=(-1.2, 1.2))
    ax.set_aspect('equal')
    ax.axis('off')

    # create empty plot
    point = ax.plot([], [], 'x', lw=2)

    def init():
        # placeholder for data
        point.set_data([], [])
        return point

    def animate(i):
        # plot dynamics as defined by i-th entry of x1 and x2
        point.set_data(x1, x2)
        return point

    ani = animation.FuncAnimation(fig, animate, x1.size,
                                  interval=dt*1000, repeat_delay=500,
                                  blit=True, init_func=init)
    return ani