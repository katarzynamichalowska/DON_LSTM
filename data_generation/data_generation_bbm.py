"""
Benjamin--Bona--Mahony equation.
"""
import numpy as np
from data_generation import sech, solve, make_trunk
import params_generate_data as p

np.random.seed(p.seed)

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

# Define the equations
f = lambda u, t: np.linalg.solve(p.I - p.D2, -np.matmul(p.D1, u + .5 * u**2) + p.g(p.x, t))
Df = lambda u: np.linalg.solve(p.I - p.D2, -np.matmul(p.D1, p.I + np.diag(u)))

# Produce u (initial condition u0) and g_u ([u1,...,uT])
u0_list, u_list = list(), list()
for i in range(p.nr_realizations):
    if i % 100==0:
        print(f"Nr samples: {i}")
    u0 = _initial_condition(p.x)
    u = solve(u0, p.t, f, Df, p.dt, p.x_points)
    u0_list.append(u0)    
    u_list.append(np.array(u)[1:])
    
u, g_u = np.array(u0_list), np.array(u_list)
g_u = g_u.reshape([p.nr_realizations, p.t_points * p.x_points])

# Trunk input (x,t)
xt = make_trunk(grid_x=p.x, grid_t=p.t)

print(f"\nData shapes:\n\t u:\t{u.shape}\n\t g_u:\t{g_u.shape}\n\t xt:\t{xt.shape}")
print(f"\nx_min:\t\t\t{xt[:,0].min()} \nx_max:\t\t\t{xt[:,0].max()} \nx_points:\t\t{p.x_points}\ndx:\t\t\t{p.dx} \nt_min:\t\t\t{xt[:,1].min()} \nt_max:\t\t\t{xt[:,1].max()} \nt_points:\t\t{p.t_points} \ndt:\t\t\t{p.dt}")

np.savez(p.filename, u=u, xt=xt, g_u=g_u)