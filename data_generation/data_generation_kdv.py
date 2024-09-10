"""
Korteweg-de Vries equation.
"""
import numpy as np
from data_generation import Datagen

class KdV(Datagen):
    def __init__(self, x_max, x_points, t_max, t_points, gamma, eta):
        super().__init__(x_max, x_points, t_max, t_points)
        self.gamma = gamma
        self.eta = eta
    
    def _initial_condition(self, grid_x):
        M = self.x_points
        P = int((grid_x[-1]-grid_x[0]) * M/(M - 1))
        k1, k2 = np.random.uniform(0.3, 0.7, 2) # height
        d1, d2 = np.random.uniform(0, 1, 2) # location
        s1 = (-6./-self.eta)*2 * k1**2 * self._sech(np.abs(k1 * ((grid_x + P/2 - P * d1) % P - P/2)))**2
        s2 = (-6./-self.eta)*2 * k2**2 * self._sech(np.abs(k2 * ((grid_x + P/2 - P * d2) % P - P/2)))**2
        u0 = s1 + s2
        u0 = np.concatenate([u0[M:], u0[:M]], axis=-1)
        return u0

    def _f(self, u, t):
        return -np.matmul(self.D1, .5* self.eta * u**2 + self.gamma**2 * np.matmul(self.D2, u)) + self._g(self.x, t)

    def _Df(self, u):
        return -np.matmul(self.D1, self.eta * np.diag(u) + self.gamma**2 * self.D2)