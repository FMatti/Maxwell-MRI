import scipy.sparse.linalg
import numpy as np
import matplotlib.pyplot as plt

def solve_eigenproblem(K, M, a=-np.inf, b=np.inf, k=10, valid_indices=None, v0=None):

    if valid_indices is not None:
        K = K[valid_indices, :][:, valid_indices]
        M = M[valid_indices, :][:, valid_indices]
        if v0 is not None:
            v0 = v0[valid_indices]
    
    if a == -np.inf or b == np.inf:
        sigma = None
    else:
        sigma = (a + b) / 2.0
        
    eigvals, eigvecs = scipy.sparse.linalg.eigsh(K, k=k, M=M, sigma=sigma, v0=v0)
    eigvals = np.sqrt(eigvals)
    
    eigvals_in_ab = [e1 for e1 in eigvals if a <= e1 and e1 <= b]
    eigvecs_in_ab = [e2 for e1, e2 in zip(eigvals, eigvecs.T) if a <= e1 and e1 <= b]

    if len(eigvals_in_ab) == k:
        print(f'WARNING: Found exactly {k} eigenvalues within [{a}, {b}].')
        print('Increase parameter "k" to make sure all eigenvalues are found.')
    
    return eigvals_in_ab, eigvecs_in_ab

def plot_L2_norms(ax, THMP, omegas, **kwargs):
    L2_norms = np.empty(len(omegas))
    for i, omega in enumerate(omegas):
        THMP.solve(omega)
        L2_norms[i] = THMP.compute_solution_norm()
    ax.plot(omegas, L2_norms, **kwargs)
    ax.set_yscale('log')
    
def plot_analytical_eigenfrequencies(ax, THMP, a, b, **kwargs):
    eigfreqs = THMP.get_analytical_eigenfrequencies(a, b)
    ax.vlines(eigfreqs, ymin=0, ymax=1, **kwargs)
    
def plot_numerical_eigenfrequencies(ax, THMP, a, b, **kwargs):
    K = THMP.get_K()
    M = THMP.get_M()
    valid_indices = THMP.get_valid_indices()
    eigfreqs, _ = solve_eigenproblem(THMP.get_K(), THMP.get_M(), a=a, b=b, k=50, valid_indices=THMP.get_valid_indices(), v0=THMP.get_N())
    ax.vlines(eigfreqs, ymin=0, ymax=1, **kwargs)
    
def vec_dot_N(A_vec, THMP):
    A_vec_inserted = THMP.insert_boundary_values(A_vec)
    return np.inner(A_vec_inserted, THMP.get_N())

def plot_eigvecs_dot_N(ax, THMP, a, b, **kwargs):
    eigvals, eigvecs = solve_eigenproblem(THMP.get_K(), THMP.get_M(), a=a, b=b, k=50, valid_indices=THMP.get_valid_indices(), v0=THMP.get_N())
    dotproducts = [vec_dot_N(vec, THMP) for vec in eigvecs]    
    ax.bar(eigvals, np.abs(dotproducts), **kwargs)
    ax.set_yscale('log')
   