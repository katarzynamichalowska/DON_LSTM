import numpy as np
from scipy.sparse import spdiags
import numpy.linalg as la

def sech(i): 
    return 1/np.cosh(i) 

def midpoint_method(u,un,t,f,Df,dt,M,tol,max_iter):
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

def difference_matrices(P=20, M=100):
    dx = P/M
    e = np.ones(M) # unit vector of length M
    # 1st order central difference matrix:
    D1 = .5/dx*spdiags([e,-e,e,-e], np.array([-M+1,-1,1,M-1]), M, M).toarray()
    # 2nd order central difference matrix:
    D2 = 1/dx**2*spdiags([e,e,-2*e,e,e], np.array([-M+1,-1,0,1,M-1]), M, M).toarray()
    return D1, D2

def grid(P=20, M=100):
    '''
    Makes array x from 0 to P-dx, with M number of elements. dx is the size between elements.
    '''
    dx = P/M
    x = np.linspace(0, P-dx, M)
    return x, dx

def make_trunk(grid_x, grid_t, remove_first_ts=True):
    # Trunk input (x,t)
    if remove_first_ts:
        x_col = np.tile(grid_x, grid_t.shape[0]-1)
        t_col = np.repeat(grid_t[1:], grid_x.shape[0])
    else:
        x_col = np.tile(grid_x, grid_t.shape[0])
        t_col = np.repeat(grid_t, grid_x.shape[0])
    xt = np.stack([x_col, t_col]).T

    return xt

def solve(u0, grid_t, f, Df, dt, M):
    u = np.zeros([grid_t.shape[0], u0.shape[-1]])
    u[0, :] = u0
    for i, t_step in enumerate(grid_t[:-1]):
        u[i+1,:] = midpoint_method(u[i,:], u[i,:], grid_t[i], f, Df, dt, M, 1e-12, 5)
        
    return u    
    