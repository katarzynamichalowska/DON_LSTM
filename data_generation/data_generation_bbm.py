"""
Benjamin--Bona--Mahony equation.
"""
import numpy as np
from data_generation import Datagen

class BBM(Datagen):  
    def _initial_condition(self, grid_x):
        M = self.x_points
        P = int((grid_x[-1] - grid_x[0]) * M/(M - 1))
        c1, c2 = np.random.uniform(1, 3, 2) # height
        d1, d2 = np.random.uniform(0, 1, 2) # location
        u0 = 0
        u0 += 3*(c1 - 1) * self._sech(1/2 * np.sqrt(1 - 1/c1) * ((grid_x + P/2 - P * d1) % P - P/2))**2
        u0 += 3*(c2 - 1) * self._sech(1/2 * np.sqrt(1 - 1/c2) * ((grid_x + P/2 - P * d2) % P - P/2))**2
        u0 = np.concatenate([u0[M:], u0[:M]], axis =- 1)
        return u0

    def _f(self, u, t):
        I = np.eye(self.x_points)
        return np.linalg.solve(I - self.D2, -np.matmul(self.D1, u + .5 * u**2) + self._g(self.x, t))

    def _Df(self, u):
        I = np.eye(self.x_points)
        return np.linalg.solve(I - self.D2, -np.matmul(self.D1, I + np.diag(u)))