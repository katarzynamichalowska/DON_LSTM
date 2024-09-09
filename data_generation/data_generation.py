import numpy as np
from scipy.sparse import spdiags
import numpy.linalg as la

def make_xt(x_len, t_len, x_2d=False):
    """
    Makes a 2D trunk input of the form:
    x: (x1, x2...xn, x1, x2...xn...xn)
    t: (t1, t1...t1, t2, t2...t2...tn)
    """
    if x_2d:
        x_len = int(np.sqrt(x_len))

    x = np.array(range(1, x_len+1))
    t = np.array(range(1, t_len+1))

    if x_2d:
        x_col1 = np.tile(np.repeat(x, x.shape[0]), t.shape[0])
        x_col2 = np.tile(np.tile(x, x.shape[0]), t.shape[0])
        t_col = np.repeat(t, x.shape[0]**2)
        xt = np.stack([t_col, x_col1, x_col2]).T
    else:
        x_col = np.tile(x, t.shape[0])
        t_col = np.repeat(t, x.shape[0])
        xt = np.stack([x_col, t_col]).T
    
    return xt

def sech(i): 
    return 1/np.cosh(i) 

def solver_schema(dx, M, gamma, eta):
    """
    Schema for the solver (or something like that :)))).
    """
    a, b, nu = 0., 0., 0.
    e = np.ones(M)
    deltacx = .5/dx*spdiags([e,-e,e,-e], np.array([-M+1,-1,1,M-1]), M, M)
    delta2cx = 1/dx**2*spdiags([e,e,-2*e,e,e], np.array([-M+1,-1,0,1,M-1]), M, M)
    deltafx = 1/dx*spdiags([e,-e,e], np.array([-M+1,0,1]), M, M)
    damping = spdiags(a*e, 0, M, M)
    D1 = deltacx.toarray() # Couldn't get sparse matrix to work
    D2 = delta2cx.toarray()
    Dp = deltafx.toarray()
    A = damping.toarray()
    f = lambda u: np.matmul(D1-A,.5*eta*u**2 + gamma**2*np.matmul(D2,u)) + nu*np.matmul(D2,u)# + np.sin(u)
    Df = lambda u: np.matmul(D1-A,eta*np.diag(u) + gamma**2*D2) + nu*D2 # - np.diag(np.cos(u))
    
    return f, Df

def random_initial_condition(x_length, x_points, eta=6.):
    """
    Produce a random initial condition with two waves.
    """
    
    dx = x_length/x_points # x_res
    x = np.linspace(0, x_length-dx, x_points) # do not include endpoint, because of periodicity
    k1, k2 = np.random.uniform(0.5, 2.0, 2)
    d1 = np.random.uniform(0.2, 0.3, 1)
    d2 = d1 + np.random.uniform(0.2, 0.5, 1)
    u0 = 0
    u0 += (-6./-eta)*2 * k1**2 * sech(k1 * (x-x_length*d1))**2
    u0 += (-6./-eta)*2 * k2**2 * sech(k2 * (x-x_length*d2))**2
    u0 = np.concatenate([u0[x_points:], u0[:x_points]], axis=-1)
    
    return u0

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

def solve_kdv(u0, grid_x, grid_t, gamma=1., eta=6.):
    
    M, N = len(grid_x), len(grid_t)
    dx = (grid_x.max()/(M-1))
    x_max = grid_x.max()+dx
    dt = (grid_t.max()/(N-1))

    D1, D2 = difference_matrices(x_max, M)
    g = lambda x, t: 0
    f = lambda u, t: -np.matmul(D1, .5*eta*u**2 + gamma**2*np.matmul(D2,u)) + g(grid_x, grid_t)
    Df = lambda u: -np.matmul(D1, eta*np.diag(u) + gamma**2*D2)
    
    u = np.zeros([grid_t.shape[0], u0.shape[-1]])
    u[0, :] = u0
    for i, t_step in enumerate(grid_t[:-1]):
        u[i+1,:] = midpoint_method(u[i,:], u[i,:], grid_t[i], f, Df, dt, M, 1e-12, 5)
        
    return u

def difference_matrices_fb(P=20, M=100):
    dx = P/M
    e = np.ones(M) # unit vector of length M
    # 1st order forward difference matrix:
    D1f = 1/dx*spdiags([e,-e,e], np.array([-M+1,0,1]), M, M).toarray()
    # 1st order backward difference matrix:
    D1b = 1/dx*spdiags([-e,e,-e], np.array([-1,0,M-1]), M, M).toarray()
    return D1f, D1b

def solve_bbm(u0, grid_x, grid_t):
    
    M, N = len(grid_x), len(grid_t)
    dx = (grid_x.max()/(M-1))
    x_max = grid_x.max()+dx
    dt = (grid_t.max()/(N-1))
    
    D1, D2 = difference_matrices(x_max, M)
    I = np.eye(M)
    
    g = lambda x, t: 0
    f = lambda u, t: np.linalg.solve(I-D2, -np.matmul(D1, u + .5*u**2) + g(grid_x, grid_t))
    Df = lambda u: np.linalg.solve(I-D2, -np.matmul(D1, I + np.diag(u)))
    
    u = np.zeros([grid_t.shape[0], u0.shape[-1]])
    u[0, :] = u0
    for i, t_step in enumerate(grid_t[:-1]):
        u[i+1,:] = midpoint_method(u[i,:], u[i,:], grid_t[i], f, Df, dt, M, 1e-12, 5)
        
    return u

def solve(u0, grid_x, grid_t, f, Df):
    M, N = len(grid_x), len(grid_t)
    dx = (grid_x.max()/(M-1))
    x_max = grid_x.max()+dx
    dt = (grid_t.max()/(N-1))
    
    D1, D2 = difference_matrices(x_max, M)

    u = np.zeros([grid_t.shape[0], u0.shape[-1]])
    u[0, :] = u0
    for i, t_step in enumerate(grid_t[:-1]):
        u[i+1,:] = midpoint_method(u[i,:], u[i,:], grid_t[i], f, Df, dt, M, 1e-12, 5)
        
    return u    
    