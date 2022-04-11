import scipy.sparse.linalg
import numpy as np
import fenics as fen
import matplotlib.pyplot as plt
import time

def solve_eigenproblem(K, M, a=-np.inf, b=np.inf, k=10, valid_indices=None, v0=None, timer=False):

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

def plot_L2_norms(ax, THMP, omegas, VS, **kwargs):
    L2_norms = np.empty(len(omegas))
    for i, omega in enumerate(omegas):
        THMP.solve(omega)
        A = THMP.get_solution(tonumpy=True, trace=VS.get_trace())
        L2_norms[i] = VS.norm(A[0])
    ax.plot(omegas, L2_norms, **kwargs)
    ax.set_yscale('log')
    
def plot_analytical_eigenfrequencies(ax, THMP, a, b, **kwargs):
    eigfreqs = THMP.get_analytical_eigenfrequencies(a, b)
    ax.vlines(eigfreqs, ymin=0, ymax=1, **kwargs)
    
def plot_numerical_eigenfrequencies(ax, THMP, a, b, k=50, timer=False, **kwargs):
    K = THMP.get_K()
    M = THMP.get_M()
    valid_indices = THMP.get_valid_indices()
    if timer:
        eigfreqs, _, dt = solve_eigenproblem(THMP.get_K(), THMP.get_M(), a=a, b=b, k=k, valid_indices=THMP.get_valid_indices(), v0=THMP.get_N(), timer=True)
    else:
        eigfreqs, _ = solve_eigenproblem(THMP.get_K(), THMP.get_M(), a=a, b=b, k=k, valid_indices=THMP.get_valid_indices(), v0=THMP.get_N())
    ax.vlines(eigfreqs, ymin=0, ymax=1, **kwargs)
    if timer:
        return dt
    
def plot_surrogate_L2_norms(ax, THMP, S, omegas, VS, **kwargs):
    THMP.solve(np.linspace(np.min(omegas), np.max(omegas), S))
    THMP.compute_surrogate(VS)
    L2_norms = np.empty(len(omegas))
    for i, omega in enumerate(omegas):
        L2_norms[i] = VS.norm(THMP.RI(omega))
    ax.plot(omegas, L2_norms, **kwargs)
    ax.set_yscale('log')
    
def plot_interpolatory_eigenfrequencies(ax, THMP, a, b, S, VS, timer=False, **kwargs):
    omegas = np.linspace(a, b, S)
    if timer:
        t0 = time.time()
        THMP.solve(omegas)
        THMP.compute_surrogate(VS)
        eigfreqs = THMP.get_interpolatory_eigenfrequencies()
        dt = time.time() - t0
    else:
        THMP.solve(omegas)
        THMP.compute_surrogate(VS)
        eigfreqs = THMP.get_interpolatory_eigenfrequencies()
    real_eigfreqs = np.real(eigfreqs[np.isreal(eigfreqs)])
    ax.vlines(real_eigfreqs, ymin=0, ymax=1, **kwargs)
    if timer:
        return dt

def vec_dot_N(A_vec, THMP):
    A_vec_inserted = THMP.insert_boundary_values(A_vec)
    return np.inner(A_vec_inserted, THMP.get_N())

def plot_eigvecs_dot_N(ax, THMP, a, b, **kwargs):
    eigvals, eigvecs = solve_eigenproblem(THMP.get_K(), THMP.get_M(), a=a, b=b, k=50, valid_indices=THMP.get_valid_indices(), v0=THMP.get_N())
    dotproducts = [vec_dot_N(vec, THMP) for vec in eigvecs]    
    ax.bar(eigvals, np.abs(dotproducts), **kwargs)
    ax.set_yscale('log')

def gram_schmidt(E, VS):
    """M-orthonormalize the columns of a matrix E"""
    E[0] /= VS.norm(E[0])
    for i in range(1, E.shape[0]):
        for j in range(i):
            E[i] -= VS.inner_product(E[j], E[i]) * E[j]
        E[i] /= VS.norm(E[i])

def get_orthonormal_matrix(shape, VS, seed=0):
    """Produce list of N orthonormal elements"""
    np.random.seed(seed)
    n1, n2 = shape
    E = np.random.randn(n1, n2)
    gram_schmidt(E, VS)
    return E

def householder_triangularization(A_, VS):
    """Compute the matrix R of a QR-decomposition of A"""
    A = A_.copy()
    
    N = A.shape[0]
    E = get_orthonormal_matrix(A.shape, VS)
    R = np.zeros((N, N))

    for k in range(N):
        R[k, k] = VS.norm(A[k])

        alpha = VS.inner_product(E[k], A[k])
        if abs(alpha) > 1e-17:
            E[k] *= - alpha / abs(alpha)

        v = R[k, k] * E[k] - A[k]
        for j in range(k):
            v -= VS.inner_product(E[j], v) * E[j]

        sigma = VS.norm(v)
        if abs(sigma) > 1e-17:
            v /= sigma
        else:
            v = E[k]

        for j in range(k+1, N):
            A[j] -= 2 * v * VS.inner_product(v, A[j])
            R[k, j] = VS.inner_product(E[k], A[j])
            A[j] -= E[k] * R[k, j]

    return R
