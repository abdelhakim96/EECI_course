def rk4step(x0, u0, ode, h):

    k1 = ode(x0, u0)
    k2 = ode(x0 + h/2 * k1, u0)
    k3 = ode(x0 + h/2 * k2, u0)
    k4 = ode(x0 + h * k3, u0)
    x_next = x0 + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    return x_next
