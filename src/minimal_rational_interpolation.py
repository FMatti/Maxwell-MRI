# -*- coding: utf-8 -*-

import numpy as np

from .rational_function import RationalFunction

class MinimalRationalInterpolation(object):
    """
    Minimal rational interpolation.

    Members
    -------
    VS : VectorSpace
        Vector space object.
    RI : RationalFunction
        The rational interpolant in barycentric coordinates.
    R : None or np.ndarray
        Upper triangular matrix (N_R x N_R) obtained from the Householder
        triangularization of the first N_R columns in A_.
    E : np.ndarray
        Orthonormal matrix created in Householder triangularization.
    V : np.ndarray
        Householder matrix created in Householder triangularization.
    sv : np.ndarray
        Singular values of the triangular matrix R.

    Methods
    -------
    compute_surrogate(snapshots, omegas) : np.ndarray, np.ndarray -> None
        Compute the rational interpolation surrogate for a snapshot matrix.
    get_surrogate() : None -> RationalFunction
        Returns the computed rational interpolation surrogate.
    get_interpolatory_eigenfrequencies() : None -> np.ndarray
        Returns the eigenfrequencies determined with the surrogate.

    References
    ----------
    [1] Pradovera D. and Nobile F.: Frequency-domain non-intrusive greedy
        Model Order Reduction based on minimal rational approximation
    [2] Giraud L. and Langou J. and Rozloznk M.: On the round-off error
        analysis of the Gram-Schmidt algorithm with reorthogonalization.
        URL: https://www.cerfacs.fr/algor/reports/2002/TR_PA_02_33.pdf
    [3] Lloyd N. Trefethen: Householder triangularization of a quasimatrix.
        IMA Journal of Numerical Analysis (2008). DOI: 10.1093/imanum/dri017
    """
    def __init__(self, VS):
        self.VS = VS
        self.RI = None
        self.R = None
        self.E = None
        self.V = None
        self.sv = None
        
    def _gram_schmidt(self, E, k=None):
        """M-orthonormalize the (k last) rows of a matrix E"""
        if k is None or k == E.shape[0]:
            k = E.shape[0]
            E[0] /= self.VS.norm(E[0])
        for i in range(E.shape[0]-k, E.shape[0]):
            E[i] -= E[:i].T.dot(self.VS.inner_product(E[:i], E[i]))
            E[i] /= self.VS.norm(E[i])
            # Repeat orthonormalization. Mantra: "Twice is enough" [2]
            E[i] -= E[:i].T.dot(self.VS.inner_product(E[:i], E[i]))
            E[i] /= self.VS.norm(E[i])

    def _set_orthonormal_matrix(self, shape):
        """Produce (extension of) orthonormal matrix with given shape"""
        n1, n2 = shape
        if self.E is None or self.E.shape[1] != n2:
            self.E = np.random.randn(n1, n2)
        else:
            # Extend orthonormal matrix E to orthonormal matrix with given shape
            n1 -= self.E.shape[0]
            self.E = np.r_[self.E, np.random.randn(n1, n2)]
        self._gram_schmidt(self.E, n1)

    def _householder_triangularization(self, A_):
        """(Sequentially) compute the upper triangular matrix of a QR-decomposition
        of the snapshot matrix A of a time-harmonic Maxwell problem."""
        A = A_.copy()
        N_A = A.shape[0]

        # Declare matrix R or extend it with zeros if it already exists
        if self.R is None:
            N_R = 0
            self.R = np.zeros((N_A, N_A)) 
        else:
            N_R = self.R.shape[0]
            self.R = np.pad(self.R, (0, N_A-N_R), mode='constant')

        # Get (or extend) an orthonormal matrix of the same shape as snapshot matrix
        self._set_orthonormal_matrix(A.shape)

        # Declare matrix V for Householder vectors or extend it if it already exists
        if self.V is None:
            self.V = np.empty((N_A, A.shape[1]))
        else:
            self.V = np.pad(self.V, ((0, N_A-N_R), (0, 0)), mode='constant')

        for j in range(N_R, N_A):
            # Apply the reflection to j-th snapshot
            for k in range(j):
                A[j] -= 2 * self.V[k] * self.VS.inner_product(self.V[k], A[j])
                self.R[k, j] = self.VS.inner_product(self.E[k], A[j])
                A[j] -= self.E[k] * self.R[k, j]

            self.R[j, j] = self.VS.norm(A[j])

            # Modify E to take account of sign
            alpha = self.VS.inner_product(self.E[j], A[j])
            if abs(alpha) > 1e-17:
                self.E[j] *= - alpha / abs(alpha)

            # Vector defining next reflection
            self.V[j] = self.R[j, j] * self.E[j] - A[j]
            for i in range(j):
                self.V[j] -= self.VS.inner_product(self.E[i], self.V[j]) * self.E[i]

            # If zero vector, reflection is j-th column in orthonormal matrix
            sigma = self.VS.norm(self.V[j])
            if abs(sigma) > 1e-17:
                self.V[j] /= sigma
            else:
                self.V[j] = self.E[j]

    def _build_surrogate(self, snapshots, omegas, additive=False):
        """Compute the rational surrogate with pre-computed snapshots"""
        if not additive:
            self.R = None
            self.E = None
            self.V = None
        self._householder_triangularization(snapshots)
        _, self.sv, V_conj = np.linalg.svd(self.R)
        # Check stability of build [TODO]
        q = V_conj[-1, :].conj()
        P = snapshots.T * q
        self.RI = RationalFunction(omegas, q, P)

    def compute_surrogate(self, snapshots, omegas, greedy=True, tol=1e-2, n=1000):
        """Compute the rational surrogate"""
        if not greedy:
            self._build_surrogate(snapshots, omegas, additive=False)
            return

        # Take smallest and largest omegas as initial support points
        supports = [np.argmin(omegas), np.argmax(omegas)]
        is_eligible = np.ones(len(omegas), dtype=bool)
        is_eligible[supports] = False
        self._build_surrogate(snapshots[supports], omegas[supports], additive=False)

        # Greedily add support points until relative surrogate error below tolerance
        while np.any(is_eligible):
            reduced_argmin = self.RI.get_denominator_argmin(omegas[is_eligible])
            argmin = np.arange(len(omegas))[is_eligible][reduced_argmin]
            supports.append(argmin)
            is_eligible[argmin] = False
            a = snapshots[argmin]
            if self.VS.norm(a - self.RI(omegas[argmin])) <= tol*self.VS.norm(a):
                # Compute surrogate using the last snapshot before termination
                self._build_surrogate(snapshots[supports], omegas[supports], additive=True)
                break
            self._build_surrogate(snapshots[supports], omegas[supports], additive=True)

    def get_surrogate(self):
        return self.RI
    
    def get_interpolatory_eigenfrequencies(self, filtered=True, only_real=False):
        """Compute the eigenfrequencies based on the roots of the rational interpolant"""
        eigfreqs = self.RI.roots(filtered)
        if only_real:
            return np.real(eigfreqs[np.isreal(eigfreqs)])
        return eigfreqs
