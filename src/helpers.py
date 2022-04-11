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

def plot_surrogate_L2_norms(ax, THMP, supports, omegas, VS, **kwargs):
    THMP.solve(supports)
    THMP.compute_surrogate(VS)
    #THMP.compute_greedy_surrogate(VS, omegas[0], omegas[-1])
    L2_norms = np.empty(len(omegas))
    for i, omega in enumerate(omegas):
        L2_norms[i] = VS.norm(THMP.RI(omega))
    ax.plot(omegas, L2_norms, **kwargs)
    ax.set_yscale('log')

def plot_interpolatory_eigenfrequencies(ax, THMP, supports, VS, timer=False, **kwargs):
    if timer:
        t0 = time.time()
        THMP.solve(supports)
        THMP.compute_surrogate(VS)
        eigfreqs = THMP.get_interpolatory_eigenfrequencies()
        dt = time.time() - t0
    else:
        THMP.solve(supports)
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

def gram_schmidt(E, VS, k=None):
    """M-orthonormalize the (k last) rows of a matrix E"""
    if k is None or k == E.shape[0]:
        k = E.shape[0]
        E[0] /= VS.norm(E[0])
    for i in range(E.shape[0]-k, E.shape[0]):
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

def extend_orthonormal_basis(E, VS, n):
    """Extend an existing orthonormal basis by n more rows"""
    E_ext = np.r_[E, np.random.randn(n, E.shape[1])]
    gram_schmidt(E_ext, VS, n)
    return E_ext

def householder_triangularization(A_, VS, R=None, E=None, V=None, returns=False):
    """Compute the matrix R of a QR-decomposition of A"""
    A = A_.copy()
    N_A = A.shape[0]

    if E is None:
        E = get_orthonormal_matrix(A.shape, VS)
    else:
        N_E = E.shape[0]
        E = extend_orthonormal_basis(E, VS, N_A-N_E)

    if R is None:
        N_R = 0
        R = np.zeros((N_A, N_A)) 
    else:
        N_R = R.shape[0]
        R = np.pad(R, (0, N_A-N_R), mode='constant')

    if V is None:
        V = np.empty((N_A, A.shape[1]))
    else:
        V = np.pad(V, ((0, N_A-N_R), (0, 0)), mode='constant')

    for j in range(N_R, N_A):

        for k in range(j):
            A[j] -= 2 * V[k] * VS.inner_product(V[k], A[j])
            R[k, j] = VS.inner_product(E[k], A[j])
            A[j] -= E[k] * R[k, j]

        R[j, j] = VS.norm(A[j])

        alpha = VS.inner_product(E[j], A[j])
        if abs(alpha) > 1e-17:
            E[j] *= - alpha / abs(alpha)

        V[j] = R[j, j] * E[j] - A[j]
        for i in range(j):
            V[j]  -= VS.inner_product(E[i], V[j] ) * E[i]

        sigma = VS.norm(V[j])
        if abs(sigma) > 1e-17:
            V[j]  /= sigma
        else:
            V[j]  = E[j]

    if returns:
        return R, E, V
    return R
