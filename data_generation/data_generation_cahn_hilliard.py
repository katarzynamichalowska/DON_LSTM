"""
Cahn-Hilliard equation.
"""
import os
import numpy as np
from data_generation import produce_samples, grid, difference_matrices
import yaml

with open('params_datagen.yml', 'r') as file:
    p = yaml.safe_load(file)

np.random.seed(p["seed"])

filename = os.path.join(p["folder"], p["filename"])

def _initial_condition(grid_x, x_max):
    a0, a1, a2, a3 = np.random.uniform(0, .2, 4)
    k0, k1, k2, k3 = np.random.randint(1, 6, 4) * 2 # Even integers
    u0 = a0 * np.sin(k0 * np.pi / x_max * grid_x)
    u0+= a1 * np.cos(k1 * np.pi / x_max * grid_x) 
    u0+= a2 * np.sin(k2 * np.pi / x_max * grid_x) 
    u0+= a3 * np.cos(k3 * np.pi / x_max * grid_x)
    return u0

# Make spatial and temporal grids
dx, dt = p["x_max"]/p["x_points"], p["t_max"]/p["t_points"]

x, dx = grid(p["x_max"], p["x_points"])
t, dt = grid(p["t_max"] + dt, p["t_points"] + 1)          # Added dt and 1 to have t_points in g_u

D1, D2 = difference_matrices(p["x_max"], p["x_points"])

g = lambda x, t: 0
I = np.eye(p["x_points"])

# Define the equations
f = lambda u, t: np.matmul(D2, p["nu"] * u + p["alpha"] * u**3 + p["mu"] * np.matmul(D2, u)) + g(x, t)
Df = lambda u: np.matmul(D2, p["nu"] * I + 3 * p["alpha"] * np.diag(u**2) + p["mu"] * D2)

# Produce u (initial condition u0) and g_u ([u1,...,uT])
u0 = _initial_condition(x, p["x_max"])
produce_samples(u0, p["nr_realizations"], x, p["x_points"], dx, t, p["t_points"], dt, f, Df, filename)
