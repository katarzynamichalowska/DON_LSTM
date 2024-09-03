"""
Korteweg-de Vries equation.
"""
import os
import numpy as np
from data_generation import grid, initial_condition_kdv, solve_kdv, make_trunk

seed = 9
np.random.seed(seed)

nr_realizations = 100
folder = "./data"
filename = "data_kdv.npz"
filename = os.path.join(folder, filename)

# Parameters
gamma, eta = 1., 6.
x_max, x_points = 10, 100                   # The length (width/period) and number of points in the spatial domain
t_max, t_points = 5, 200                    # Total seconds, nr points in g_u
dx, dt = x_max/x_points, t_max/t_points

# Make spatial and temporal grids
x, dx = grid(x_max, x_points)
t, dt = grid(t_max+dt, t_points+1)          # Added dt and 1 to have t_points in g_u


# Produce u (initial condition u0) and g_u ([u1,...,uT])
u0_list, u_list = list(), list()
for i in range(nr_realizations):
    if i%100==0:
        print(f"Nr samples: {i}")
    u0 = initial_condition_kdv(x, eta=eta)
    u = solve_kdv(u0, x, t, gamma=gamma, eta=eta)
    u0_list.append(u0)    
    u_list.append(np.array(u)[1:])
    
u, g_u = np.array(u0_list), np.array(u_list)
g_u = g_u.reshape([nr_realizations, t_points * x_points])

# Trunk input (x,t)
xt = make_trunk(grid_x=x, grid_t=t)

print(f"\nData shapes:\n\t u:\t{u.shape}\n\t g_u:\t{g_u.shape}\n\t xt:\t{xt.shape}")
print(f"\nx_min:\t\t\t{xt[:,0].min()} \nx_max:\t\t\t{xt[:,0].max()} \nx_points:\t\t{x_points}\ndx:\t\t\t{dx} \nt_min:\t\t\t{xt[:,1].min()} \nt_max:\t\t\t{xt[:,1].max()} \nt_points:\t\t{t_points} \ndt:\t\t\t{dt}")

np.savez(filename, u=u, xt=xt, g_u=g_u)