"""
Korteweg-de Vries equation.
"""
import numpy as np
from data_generation import sech, produce_samples
import params_generate_data as p

np.random.seed(p.seed)

def _initial_condition(grid_x, eta=6.):
    M = p.x_points
    P = int((grid_x[-1]-grid_x[0]) * M/(M - 1))
    k1, k2 = np.random.uniform(0.3, 0.7, 2) # height
    d1, d2 = np.random.uniform(0, 1, 2) # location
    s1 = (-6./-eta)*2 * k1**2 * sech(np.abs(k1 * ((grid_x + P/2 - P * d1) % P - P/2)))**2
    s2 = (-6./-eta)*2 * k2**2 * sech(np.abs(k2 * ((grid_x + P/2 - P * d2) % P - P/2)))**2
    u0 = s1 + s2
    u0 = np.concatenate([u0[M:], u0[:M]], axis=-1)
    return u0

# Define the equations
f = lambda u, t: -np.matmul(p.D1, .5* p.eta * u**2 + p.gamma**2 * np.matmul(p.D2, u)) + p.g(p.x, t)
Df = lambda u: -np.matmul(p.D1, p.eta * np.diag(u) + p.gamma**2 * p.D2)

# Produce u (initial condition u0) and g_u ([u1,...,uT])
u0 = _initial_condition(p.x, eta=p.eta)
produce_samples(u0, p.nr_realizations, p.x, p.x_points, p.dx, p.t, p.t_points, p.dt, f, Df, p.filename)