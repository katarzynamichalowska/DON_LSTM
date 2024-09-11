"""
Cahn-Hilliard equation.
"""
import numpy as np
from data_generation import Datagen

class CahnHill(Datagen):
    def __init__(self, x_max, x_points, t_max, t_points, nu, alpha, mu):
        super().__init__(x_max, x_points, t_max, t_points)
        self.nu = nu
        self.alpha = alpha
        self.mu = mu
    
    def _initial_condition(self, grid_x):
        a0, a1, a2, a3 = np.random.uniform(0, .2, 4)
        k0, k1, k2, k3 = np.random.randint(1, 6, 4) * 2 # Even integers
        u0 = a0 * np.sin(k0 * np.pi / self.x_max * grid_x)
        u0+= a1 * np.cos(k1 * np.pi / self.x_max * grid_x) 
        u0+= a2 * np.sin(k2 * np.pi / self.x_max * grid_x) 
        u0+= a3 * np.cos(k3 * np.pi / self.x_max * grid_x)
        return u0

    def _f(self, u, t):
        return np.matmul(self.D2, self.nu * u + self.alpha * u**3 + self.mu * np.matmul(self.D2, u)) + self._g(self.x, t)

    def _Df(self, u):
        I = np.eye(self.x_points)
        return np.matmul(self.D2, self.nu * I + 3 * self.alpha * np.diag(u**2) + self.mu * self.D2)