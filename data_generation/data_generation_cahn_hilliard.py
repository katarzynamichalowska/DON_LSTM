"""
Cahn-Hilliard equation.
"""
import numpy as np
from data_generation import produce_samples
import params_generate_data as p

np.random.seed(p.seed)

def _initial_condition(grid_x, x_max):
    a0, a1, a2, a3 = np.random.uniform(0, .2, 4)
    k0, k1, k2, k3 = np.random.randint(1, 6, 4) * 2 # Even integers
    u0 = a0 * np.sin(k0 * np.pi / x_max * grid_x)
    u0+= a1 * np.cos(k1 * np.pi / x_max * grid_x) 
    u0+= a2 * np.sin(k2 * np.pi / x_max * grid_x) 
    u0+= a3 * np.cos(k3 * np.pi / x_max * grid_x)
    return u0



# Define the equations
f = lambda u, t: np.matmul(p.D2, p.nu * u + p.alpha * u**3 + p.mu * np.matmul(p.D2, u)) + p.g(p.x, t)
Df = lambda u: np.matmul(p.D2, p.nu * p.I + 3 * p.alpha * np.diag(u**2) + p.mu * p.D2)

# Produce u (initial condition u0) and g_u ([u1,...,uT])
u0 = _initial_condition(p.x, p.x_max)
produce_samples(u0, p.nr_realizations, p.x, p.x_points, p.dx, p.t, p.t_points, p.dt, f, Df, p.filename)
