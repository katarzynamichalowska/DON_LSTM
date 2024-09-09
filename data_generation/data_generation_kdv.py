"""
Korteweg-de Vries equation.
"""
import os
import numpy as np
from data_generation import sech, produce_samples, grid, difference_matrices
import params_generate_data as p

np.random.seed(p.seed)

filename = os.path.join(p.folder, p.filename)

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

# Make spatial and temporal grids
x, dx = grid(p.x_max, p.x_points)
t, dt = grid(p.t_max + p.dt, p.t_points + 1)          # Added dt and 1 to have t_points in g_u

D1, D2 = difference_matrices(p.x_max, p.x_points)

g = lambda x, t: 0

# Define the equations
f = lambda u, t: -np.matmul(D1, .5* p.eta * u**2 + p.gamma**2 * np.matmul(D2, u)) + g(x, t)
Df = lambda u: -np.matmul(D1, p.eta * np.diag(u) + p.gamma**2 * D2)

# Produce u (initial condition u0) and g_u ([u1,...,uT])
u0 = _initial_condition(x, eta=p.eta)
produce_samples(u0, p.nr_realizations, x, p.x_points, p.dx, t, p.t_points, p.dt, f, Df, filename)