"""
Korteweg-de Vries equation.
"""
import os
import numpy as np
from data_generation import grid, sech, solve, difference_matrices, make_trunk

seed = 9
np.random.seed(seed)

nr_realizations = 1000
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

D1, D2 = difference_matrices(x_max, x_points)

def _initial_condition(grid_x, eta=6.):
    M = x_points
    
    P = int((grid_x[-1]-grid_x[0])*M/(M-1))
    k1, k2 = np.random.uniform(0.3, 0.7, 2) # height
    d1, d2 = np.random.uniform(0, 1, 2) # location
    s1 = (-6./-eta)*2 * k1**2 * sech(np.abs(k1 * ((grid_x+P/2-P*d1) % P - P/2)))**2
    s2 = (-6./-eta)*2 * k2**2 * sech(np.abs(k2 * ((grid_x+P/2-P*d2) % P - P/2)))**2
    u0 = s1 + s2
    u0 = np.concatenate([u0[M:], u0[:M]], axis=-1)

    return u0

# Define the equations
g = lambda x, t: 0
f = lambda u, t: -np.matmul(D1, .5*eta*u**2 + gamma**2*np.matmul(D2,u)) + g(x, t)
Df = lambda u: -np.matmul(D1, eta*np.diag(u) + gamma**2*D2)

# Produce u (initial condition u0) and g_u ([u1,...,uT])
u0_list, u_list = list(), list()
for i in range(nr_realizations):
    if i%100==0:
        print(f"Nr samples: {i}")
    u0 = _initial_condition(x, eta=eta)
    u = solve(u0, t, f, Df, dt, x_points)
    u0_list.append(u0)    
    u_list.append(np.array(u)[1:])
    
u, g_u = np.array(u0_list), np.array(u_list)
g_u = g_u.reshape([nr_realizations, t_points * x_points])

# Trunk input (x,t)
xt = make_trunk(grid_x=x, grid_t=t)

print(f"\nData shapes:\n\t u:\t{u.shape}\n\t g_u:\t{g_u.shape}\n\t xt:\t{xt.shape}")
print(f"\nx_min:\t\t\t{xt[:,0].min()} \nx_max:\t\t\t{xt[:,0].max()} \nx_points:\t\t{x_points}\ndx:\t\t\t{dx} \nt_min:\t\t\t{xt[:,1].min()} \nt_max:\t\t\t{xt[:,1].max()} \nt_points:\t\t{t_points} \ndt:\t\t\t{dt}")

np.savez(filename, u=u, xt=xt, g_u=g_u)