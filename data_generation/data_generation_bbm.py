"""
Benjamin--Bona--Mahony equation.
"""
import os
import numpy as np
from data_generation import sech, produce_samples, grid, difference_matrices
import params_generate_data as p

np.random.seed(p.seed)

filename = os.path.join(p.folder, p.filename)

def _initial_condition(grid_x):
    M = p.x_points
    P = int((grid_x[-1] - grid_x[0]) * M/(M - 1))
    c1, c2 = np.random.uniform(1, 3, 2) # height
    d1, d2 = np.random.uniform(0, 1, 2) # location
    u0 = 0
    u0 += 3*(c1 - 1) * sech(1/2 * np.sqrt(1 - 1/c1) * ((grid_x + P/2 - P * d1) % P - P/2))**2
    u0 += 3*(c2 - 1) * sech(1/2 * np.sqrt(1 - 1/c2) * ((grid_x + P/2 - P * d2) % P - P/2))**2
    u0 = np.concatenate([u0[M:], u0[:M]], axis =- 1)
    return u0

# Make spatial and temporal grids
x, dx = grid(p.x_max, p.x_points)
t, dt = grid(p.t_max + p.dt, p.t_points + 1)          # Added dt and 1 to have t_points in g_u

D1, D2 = difference_matrices(p.x_max, p.x_points)

g = lambda x, t: 0
I = np.eye(p.x_points)

# Define the equations
f = lambda u, t: np.linalg.solve(I - D2, -np.matmul(D1, u + .5 * u**2) + g(x, t))
Df = lambda u: np.linalg.solve(I - D2, -np.matmul(D1, I + np.diag(u)))

# Produce u (initial condition u0) and g_u ([u1,...,uT])
u0 = _initial_condition(x)
produce_samples(u0, p.nr_realizations, x, p.x_points, p.dx, t, p.t_points, p.dt, f, Df, filename)