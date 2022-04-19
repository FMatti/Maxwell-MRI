# -*- coding: utf-8 -*-

import numpy as np
import fenics as fen
import scipy.sparse

from .rational_function import RationalFunction

class MinimalRationalInterpolation(object):
    """
    Minimal rational interpolator for time-harmonic Maxwell problems.

    Members
    -------
    A_ : np.ndarray
        Snapshot matrix.
    VS : VectorSpace
        Vector space object.
    R : None or np.ndarray
        Upper triangular matrix (N_R x N_R) obtained from the Householder
        triangularization of the first N_R columns in A_.
    E : np.ndarray
        Orthonormal matrix created in Householder triangularization.
    V : np.ndarray
        Householder matrix created in Householder triangularization.

    Methods
    -------
    
    References
    ----------
    [1] Pradovera D. and Nobile F.: Frequency-domain non-intrusive greedy
        Model Order Reduction based on minimal rational approximation
    [2] Giraud L. and Langou J. and Rozloznk M.: On the round-off error
        analysis of the Gram-Schmidt algorithm with reorthogonalization.
        URL: https://www.cerfacs.fr/algor/reports/2002/TR_PA_02_33.pdf
    """

    def __init__(self, THMP, VS):
        self.THMP = THMP
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
            for j in range(i):
                E[i] -= self.VS.inner_product(E[j], E[i]) * E[j]
            E[i] /= self.VS.norm(E[i])
            # Repeat orthonormalization: "Twice is enough" [2]
            for j in range(i):
                E[i] -= self.VS.inner_product(E[j], E[i]) * E[j]
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
        """
        (Sequentially) compute the upper triangular matrix of a QR-decomposition
        of the snapshot matrix A of a time-harmonic Maxwell problem. 

        References
        ----------
        [1] Lloyd N. Trefethen: Householder triangularization of a quasimatrix.
            IMA Journal of Numerical Analysis (2008). DOI: 10.1093/imanum/dri017
        """
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

    def _build_surrogate(self, additive=False):
        """Compute the rational surrogate with pre-computed snapshots"""
        A = self.THMP.get_solution(tonumpy=True, trace=self.VS.get_trace())
        if not additive:
            self.R = None
            self.E = None
            self.V = None
        self._householder_triangularization(A)
        _, self.sv, V_conj = np.linalg.svd(self.R)
        # Check stability [TODO]
        omega = self.THMP.get_frequency()
        q = V_conj[-1, :].conj()
        P = A.T * q
        self.RI = RationalFunction(omega, q, P)

    def compute_surrogate(self, mode='snapshots', a=None, b=None, tol=1e-2, n=1000):
        """Compute the rational surrogate"""
        if mode is not 'greedy':
            self.build_surrogate()
            return
        self.THMP.solve([a, b])
        self._build_surrogate()
        samples = np.linspace(a, b, n)[1:-1]
        while len(samples) > 0:
            samples_min, index_min = self.RI.get_denominator_argmin(samples)
            self.THMP.solve(samples_min, accumulate=True)
            a = self.THMP.get_solution(tonumpy=True, trace=self.VS.get_trace())[-1]
            if self.VS.norm(a - self.RI(samples_min)) <= tol*self.VS.norm(a):
                # Compute surrogate using the last snapshot before termination
                self._build_surrogate(additive=True)
                break
            samples = np.delete(samples, index_min)
            self._build_surrogate(additive=True)

    def get_surrogate(self):
        return self.RI

    def get_vectorspace(self):
        return self.VS
    
    def get_interpolatory_eigenfrequencies(self, filtered=True):
        """Compute the eigenfrequencies based on the roots of the rational interpolant"""
        return self.RI.roots(filtered)

    def plot_surrogate_norm(self, ax, a=None, b=None, N=1000, **kwargs):
        if a is None:
            a = np.min(self.RI.get_nodes())
        if b is None:
            b = np.max(self.RI.get_nodes())
        linspace = np.linspace(a, b, N)
        ax.plot(linspace, [self.VS.norm(self.RI(x)) for x in linspace], **kwargs)
        ax.set_yscale('log')
