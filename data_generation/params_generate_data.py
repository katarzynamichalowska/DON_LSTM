seed = 9

nr_realizations = 200
folder = "./data"
filename = "data_kdv.npz"

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