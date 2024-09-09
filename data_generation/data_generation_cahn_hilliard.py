"""
Cahn-Hilliard equation.
"""

import os
import numpy as np
from data_generation import grid, make_trunk, difference_matrices, solve

seed = 9
np.random.seed(seed)

nr_realizations = 1000
folder = "data"
filename = "data_cahn_hilliard.npz"
filename = os.path.join(folder, filename)

x_max, x_points = 1, 100                    # The length (width/period) and number of points in the spatial domain
t_max, t_points = 3, 150                    # Total seconds, nr points in g_u
dx, dt = x_max/x_points, t_max/t_points

x, dx = grid(x_max, x_points)
t, dt = grid(t_max+dt, t_points+1)          # Added dt and 1 to have t_points in g_u

D1, D2 = difference_matrices(x_max, x_points)

# Parameters (setup1):
nu = -.01
alpha = .01
mu = -0.00001

# Parameters (setup2):
#nu = -1
#alpha = -0.001
#mu = 1

def _initial_condition(grid_x, x_max):
    '''
    Superposition of two sine and two cosine waves with random parameters.
    '''
    a0, a1, a2, a3 = np.random.uniform(0,.2,4)
    k0, k1, k2, k3 = np.random.randint(1,6,4)*2 # Even integers
    u0 = a0 * np.sin(k0*np.pi / x_max*grid_x)
    u0+= a1 * np.cos(k1*np.pi / x_max*grid_x) 
    u0+= a2 * np.sin(k2*np.pi / x_max*grid_x) 
    u0+= a3 * np.cos(k3*np.pi / x_max*grid_x)
    
    return u0

g = lambda x, t: 0
I = np.eye(x_points)

# Define the equation
f = lambda u, t: np.matmul(D2, nu*u + alpha*u**3 + mu*np.matmul(D2,u)) + g(x, t)
Df = lambda u: np.matmul(D2, nu*I + 3*alpha*np.diag(u**2) + mu*D2)

# The energy integral
#V = lambda u: dx/2*(nu*np.dot(u,u) + 1/2*alpha*np.dot(u**2,u**2) - mu*np.dot(np.matmul(D1,u),np.matmul(D1,u)))

u0_list, u_list = list(), list()
for i in range(nr_realizations):
    
    if i%100==0:
        print(f"Nr samples: {i}")
    u0 = _initial_condition(x, x_max)

    u = solve(u0, t, f, Df, dt, x_points)
    u0_list.append(u0)    
    u_list.append(np.array(u)[1:])
    
u, g_u = np.array(u0_list), np.array(u_list)
g_u = g_u.reshape([nr_realizations, t_points * x_points])

# Trunk input (x,t)
xt = make_trunk(grid_x=x, grid_t=t)

np.savez(filename, u=u, xt=xt, g_u=g_u)
