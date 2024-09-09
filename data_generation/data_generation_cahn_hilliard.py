"""
Cahn-Hilliard equation.
"""
import numpy as np
from data_generation import make_trunk, solve
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
u0_list, u_list = list(), list()
for i in range(p.nr_realizations):
    if i % 100==0:
        print(f"Nr samples: {i}")
    u0 = _initial_condition(p.x, p.x_max)
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
