import cvxpy as cp
import numpy as np
from scipy.linalg import sqrtm

def graph_form_constants(mus,sigmas,pi):
    """Construct the affine constants for the graph form of the EVaR problem.
    """

    n = len(mus[0])
    k = len(pi)
    dim_socp = n + 2 

    #Quad cone
    F_quad = np.vstack([np.eye(n)]+[np.zeros((1,n)) for i in range(2)])
    d_quad = np.concatenate([np.zeros(n),np.array([.5,.5])])
    e_quad = np.concatenate([np.zeros(n),np.array([-.5,.5])])

    As = [(1/(2**.5))*sqrtm(sigmas[i]) for i in range(k)]
    bs = []
    for i in range(k):
        S = sigmas[i]
        v,U = np.linalg.eigh(S)
        S_half_inv = U@np.diag(1/v**.5)@U.T
        bs.append(-((2**.5)/2)*S_half_inv@mus[i])
        
    e_i_quads = [F_quad@bs[i]+e_quad+((1/2)*(mus[i]@np.linalg.inv(sigmas[i])@mus[i])-np.log(pi[i]))*d_quad for i in range(k)]

    #LSE cone
    F_LSE = np.vstack([np.zeros((1,k))]+[np.vstack([np.eye(k)[i,:],np.zeros((2,k))]) for i in range(k)])
    G_LSE = np.vstack([np.ones((1,k))]+[np.vstack([np.zeros((2,k)),np.eye(k)[i,:]]) for i in range(k)])
    d_LSE = np.concatenate([np.array([0])]+[-np.eye(3)[0] for i in range(k)])
    e_LSE = np.concatenate([np.array([-1])]+[np.eye(3)[1] for i in range(k)])

    #Final cone 
    F_K = np.vstack([np.zeros((1+3*k,n))]+[F_quad@As[i] for i in range(k)])
    G_K = np.hstack([
        np.vstack([G_LSE]+[np.zeros((dim_socp,k))]*k),
        np.vstack([F_LSE]+[np.outer(d_quad,np.eye(k)[i]) for i in range(k)])
    ])

    d_K = np.concatenate([d_LSE]+[np.zeros(dim_socp*k)])
    e_K = np.concatenate([e_LSE]+[e_i_quads[i] for i in range(k)])
    F_K_tilde = np.hstack([F_K,e_K[:,None]]) 

    return F_K_tilde,G_K,d_K

def min_EVaR_portfolio(alpha,L,mus,sigmas,pi):
    """Compute a minimal EVaR-alpha portfolio, with leverage limit L
    """

    # Problem dimensions
    n = len(mus[0])
    k = len(pi)
    dim_socp = n + 2 

    # Problem variables
    w = cp.Variable(n)
    z = cp.Variable(2*k)
    t = cp.Variable()
    delta = cp.Variable(nonneg=True)

    # The problem objective comprised the epigraph upper bound on P(w,delta) together with
    # the affine term to agree with the full EVaR expression
    obj = t - delta*np.log(alpha)
    
    # Ininitializing the basic portfolio constraints
    constraints = [cp.sum(w) == 1 ,cp.norm(w,1) <= L]

    # Getting the graph form affine constants
    F_K_tilde,G_K,d_K = graph_form_constants(mus,sigmas,pi)

    w_delta = cp.hstack((w,delta))
    x_cone = F_K_tilde@w_delta + G_K@z+t*d_K

    # Log-sum-exp cone constraints
    x_LSE = x_cone[:3*k+1]
    constraints += [x_LSE[0] <= 0]
    for i in range(k):
        x_exp_i = x_LSE[1+i*3:1+(i+1)*3]
        constraints += [cp.constraints.ExpCone(*x_exp_i)]

    # Second order cone constraints
    x_SOCP = x_cone[3*k+1:]
    for i in range(k):
        x_socp_i = x_SOCP[i*dim_socp:(i+1)*dim_socp]
        constraints += [cp.norm(x_socp_i[:-1],2) <= x_socp_i[-1]]
        
    prob = cp.Problem(cp.Minimize(obj),constraints)
    prob.solve()

    return w.value,delta.value,prob.value