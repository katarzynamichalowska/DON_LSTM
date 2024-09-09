import os
import numpy as np
from data_generation import grid, difference_matrices

seed = 9

nr_realizations = 1000
folder = "./data"
filename = "data_kdv.npz"
filename = os.path.join(folder, filename)

# Parameters
x_max, x_points = 10, 100                   # The length (width/period) and number of points in the spatial domain
t_max, t_points = 5, 200                    # Total seconds, nr points in g_u
dx, dt = x_max/x_points, t_max/t_points

# Parameters (Cahn-Hilliard):
nu = -.01
alpha = .01
mu = -0.00001

# Parameters (KdV)
gamma, eta = 1., 6.

# Make spatial and temporal grids
x, dx = grid(x_max, x_points)
t, dt = grid(t_max+dt, t_points+1)          # Added dt and 1 to have t_points in g_u

D1, D2 = difference_matrices(x_max, x_points)

g = lambda x, t: 0
I = np.eye(x_points)