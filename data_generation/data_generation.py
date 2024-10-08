import numpy as np
from scipy.sparse import spdiags
import numpy.linalg as la
from abc import ABC, abstractmethod

class Datagen(ABC):
    def __init__(self, x_max, x_points, t_max, t_points):
        self.x_max = x_max
        self.x_points = x_points

        self.t_max = t_max
        self.t_points = t_points

        self.dx, dt = x_max/x_points, t_max/t_points

        self.x, self.dx = self._grid(x_max, x_points)
        self.t, self.dt = self._grid(t_max + dt, t_points + 1)  

        self.D1, self.D2 = self._difference_matrices(x_max, x_points)

    @abstractmethod
    def _initial_condition(self, grid_x):
        pass

    @abstractmethod
    def _f(self, u, t):
        pass

    @abstractmethod
    def _Df(self, u):
        pass

    def _g(self, x, t): 
        return 0

    def _sech(self, i): 
        return 1/np.cosh(i) 

    def _midpoint_method(self, u, un, t, f, Df, dt, M, tol, max_iter):
        '''
        Integrating one step of the ODE u_t = f, from u to un,
        with the implicit midpoint method
        Using Newton's method to find un
        dt = step size
        '''
        I = np.eye(M)
        F = lambda u_hat: 1/dt*(u_hat-u) - f((u+u_hat)/2, t+.5*dt)
        J = lambda u_hat: 1/dt*I - 1/2*Df((u+u_hat)/2)
        err = la.norm(F(un))
        it = 0
        while err > tol:
            un = un - la.solve(J(un),F(un))
            err = la.norm(F(un))
            it += 1
            if it > max_iter:
                break
        return un

    def _difference_matrices(self, P=20, M=100):
        '''
        Discrete approximation of the first and second order 
        spatial derivative operators
        f = u_x^2 + u_xx
        = (d/dx u)^2 + d^2/dx^2 u
        D1 approximates d/dx, D2 approximates d^2/dx^2
        '''
        dx = P/M
        e = np.ones(M) # unit vector of length M
        # 1st order central difference matrix:
        D1 = .5/dx*spdiags([e,-e,e,-e], np.array([-M+1,-1,1,M-1]), M, M).toarray()
        # 2nd order central difference matrix:
        D2 = 1/dx**2*spdiags([e,e,-2*e,e,e], np.array([-M+1,-1,0,1,M-1]), M, M).toarray()
        return D1, D2

    def _grid(self, P=20, M=100):
        '''
        Makes array x from 0 to P-dx, with M number of elements. dx is the size between elements.
        '''
        dx = P/M
        x = np.linspace(0, P-dx, M)
        return x, dx

    def _make_trunk(self, grid_x, grid_t, remove_first_ts=True):
        # Trunk input (x,t)
        if remove_first_ts:
            x_col = np.tile(grid_x, grid_t.shape[0]-1)
            t_col = np.repeat(grid_t[1:], grid_x.shape[0])
        else:
            x_col = np.tile(grid_x, grid_t.shape[0])
            t_col = np.repeat(grid_t, grid_x.shape[0])
        xt = np.stack([x_col, t_col]).T

        return xt

    def _solve(self, u0, grid_t, f, Df, dt, M):
        '''
        Solve equations f and Df with initial condition u0 with the midpoint method.
        grid_t is the temporal grid. dt is the difference between two consecutive points in grid_t.
        M is the number of points in the spatial grid.
        '''
        u = np.zeros([grid_t.shape[0], u0.shape[-1]])
        u[0, :] = u0
        for i, t_step in enumerate(grid_t[:-1]):
            u[i+1,:] = self._midpoint_method(u[i,:], u[i,:], grid_t[i], f, Df, dt, M, 1e-12, 5)
            
        return u    
        
    def produce_samples(self, realizations, filename):
        '''
        Produce samples using spatial and temporal grid x and t, and initial conditions u0 and equations f and Df.
        '''
        u0_list, u_list = list(), list()
        for i in range(realizations):
            if i % 100==0:
                print(f"Nr samples: {i}")
            u0 = self._initial_condition(self.x)
            u = self._solve(u0, self.t, self._f, self._Df, self.dt, self.x_points)
            u0_list.append(u0)    
            u_list.append(np.array(u)[1:])
            
        u, g_u = np.array(u0_list), np.array(u_list)
        g_u = g_u.reshape([realizations, self.t_points * self.x_points])

        # Trunk input (x,t)
        xt = self._make_trunk(grid_x=self.x, grid_t=self.t)

        # Log sample descriptions
        print(f"\nData shapes:\n\t u:\t{u.shape}\n\t g_u:\t{g_u.shape}\n\t xt:\t{xt.shape}")
        print(f"\nx_min:\t\t\t{xt[:,0].min()} \nx_max:\t\t\t{xt[:,0].max()} \nx_points:\t\t{self.x_points}\ndx:\t\t\t{self.dx} \nt_min:\t\t\t{xt[:,1].min()} \nt_max:\t\t\t{xt[:,1].max()} \nt_points:\t\t{self.t_points} \ndt:\t\t\t{self.dt}")

        np.savez(filename, u=u, xt=xt, g_u=g_u)