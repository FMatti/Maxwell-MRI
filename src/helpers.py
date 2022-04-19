import scipy.sparse.linalg
import numpy as np
import fenics as fen
import matplotlib.pyplot as plt
import time

def solve_eigenproblem(K, M, a=-np.inf, b=np.inf, k=10, valid_indices=None, v0=None, timer=False):
    """Solve an eigenvalue problem of the shape K*v = a*M*v"""
    if valid_indices is not None:
        K = K[valid_indices, :][:, valid_indices]
        M = M[valid_indices, :][:, valid_indices]
        if v0 is not None:
            v0 = v0[valid_indices]

    if a == -np.inf or b == np.inf:
        sigma = None
    else:
        sigma = (a + b) / 2.0

    t0 = time.time()
    eigvals, eigvecs = scipy.sparse.linalg.eigsh(K, k=k, M=M, sigma=sigma, v0=v0)
    dt = time.time() - t0
    eigvals = np.sqrt(eigvals)

    eigvals_in_ab = [e1 for e1 in eigvals if a <= e1 and e1 <= b]
    eigvecs_in_ab = [e2 for e1, e2 in zip(eigvals, eigvecs.T) if a <= e1 and e1 <= b]

    if timer:
        return eigvals_in_ab, eigvecs_in_ab, dt

    if len(eigvals_in_ab) == k:
        print(f'WARNING: Found exactly {k} eigenvalues within [{a}, {b}].')
        print('Increase parameter "k" to make sure all eigenvalues are found.')
    
    return eigvals_in_ab, eigvecs_in_ab

def plot_solution_norms(ax, THMP, VS, **kwargs):
    """Plot the norm of the solution for a set of frequencies"""
    omegas = THMP.get_frequency()
    norms = np.empty(len(omegas))
    A = THMP.get_solution(tonumpy=True, trace=VS.get_trace())
    for i, omega in enumerate(omegas):
        norms[i] = VS.norm(A[i])
    ax.plot(omegas, norms, **kwargs)
    ax.set_yscale('log')

def plot_analytical_eigenfrequencies(ax, THMP, a, b, **kwargs):
    """Plot analytical eigenfrequencies of a time-harmonic Maxwell problem"""
    eigfreqs = THMP.get_analytical_eigenfrequencies(a, b)
    ax.vlines(eigfreqs, ymin=0, ymax=1, **kwargs)

def plot_numerical_eigenfrequencies(ax, THMP, a, b, k=50, timer=False, **kwargs):
    """Plot numerical eigenfrequencies of a time-harmonic Maxwell problem"""
    K = THMP.get_K()
    M = THMP.get_M()
    valid_indices = THMP.get_valid_indices()
    if timer:
        eigfreqs, _, dt = solve_eigenproblem(THMP.get_K(), THMP.get_M(), a=a, b=b, k=k,
                                             valid_indices=THMP.get_valid_indices(),
                                             v0=THMP.get_N(), timer=True)
    else:
        eigfreqs, _ = solve_eigenproblem(THMP.get_K(), THMP.get_M(), a=a, b=b, k=k,
                                         valid_indices=THMP.get_valid_indices(),
                                         v0=THMP.get_N())
    ax.vlines(eigfreqs, ymin=0, ymax=1, **kwargs)
    if timer:
        return dt

def plot_surrogate_norms(ax, MRI, VS=None, a=None, b=None, N=1000, **kwargs):
    """Plot vectorspace norm of surrogate at N uniform points in [a, b]"""
    if VS is None:
        VS = MRI.get_vectorspace()
    if a is None:
        a = np.min(self.RI.get_nodes())
    if b is None:
        b = np.max(self.RI.get_nodes())
    norms = np.empty(N)
    RI = MRI.get_surrogate()
    linspace = np.linspace(a, b, N)
    ax.plot(linspace, [VS.norm(MRI.RI(x)) for x in linspace], **kwargs)
    ax.set_yscale('log')

def plot_surrogate_error_norms(ax, THMP, MRI, VS, **kwargs):
    """Plot relative error norm of surrogate"""
    omegas = THMP.get_frequency()
    A = THMP.get_solution(tonumpy=True, trace=VS.get_trace())
    RI = MRI.get_surrogate()
    err = [VS.norm(A[i] - RI(x)) / VS.norm(A[i]) for i, x in enumerate(omegas)]
    ax.plot(omegas, err, **kwargs)
    ax.set_yscale('log')

    
    
    
def plot_surrogate_support_points(ax, THMP, **kwargs):
    supports = THMP.RI.get_nodes()
    ax.vlines(supports, ymin=0, ymax=1, **kwargs)

def plot_interpolatory_eigenfrequencies(ax, THMP, VS, **kwargs):
    eigfreqs = THMP.get_interpolatory_eigenfrequencies()
    real_eigfreqs = np.real(eigfreqs[np.isreal(eigfreqs)])
    ax.vlines(real_eigfreqs, ymin=0, ymax=1, **kwargs)

def vec_dot_N(A_vec, THMP):
    A_vec_inserted = THMP.insert_boundary_values(A_vec)
    return np.inner(A_vec_inserted, THMP.get_N())

def plot_eigvecs_dot_N(ax, THMP, a, b, **kwargs):
    eigvals, eigvecs = solve_eigenproblem(THMP.get_K(), THMP.get_M(), a=a, b=b, k=50, valid_indices=THMP.get_valid_indices(), v0=THMP.get_N())
    dotproducts = [vec_dot_N(vec, THMP) for vec in eigvecs]    
    ax.bar(eigvals, np.abs(dotproducts), **kwargs)
    ax.set_yscale('log')
