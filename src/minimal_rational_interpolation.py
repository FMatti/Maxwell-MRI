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
    A_ring : RationalFunction
        Constituent of the rational interpolant in barycentric coordinates.
    R : None or np.ndarray
        Upper triangular matrix (N_R x N_R) obtained from the Householder
        triangularization of the first N_R columns in A_.
    E : np.ndarray
        Orthonormal matrix created in Householder triangularization.
    V : np.ndarray
        Householder matrix created in Householder triangularization.
    supports : list
        Indices of support points used to build surrogate from snapshots.    

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
        self.A_ring = None
        self.R = None
        self.E = None
        self.V = None
        self.supports = None

    def _gram_schmidt(self, E, k=None):
        """Orthonormalize the (k last) rows of a matrix E"""
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

    def _householder_triangularization(self, A):
        """(Sequentially) compute the upper triangular matrix of a QR-decomposi-
        tion of the snapshot matrix A of a time-harmonic Maxwell problem."""

        N_A = A.shape[0]

        # Declare matrix R or extend it with zeros if it already exists
        if self.R is None:
            N_R = 0
            self.R = np.zeros((N_A, N_A)) 
        else:
            N_R = self.R.shape[0]
            self.R = np.pad(self.R, (0, N_A-N_R), mode='constant')

        # Get/extend orthonormal matrix E to the shape of snapshot matrix
        self._set_orthonormal_matrix(A.shape)

        # Declare/extend matrix V for Householder vectors
        if self.V is None:
            self.V = np.empty((N_A, A.shape[1]))
        else:
            self.V = np.pad(self.V, ((0, N_A-N_R), (0, 0)), mode='constant')

        for j in range(N_R, N_A):
            a = A[j].copy()

            # Apply the reflection to j-th snapshot
            for k in range(j):
                a -= 2 * self.VS.inner_product(self.V[k], a) * self.V[k]
                self.R[k, j] = self.VS.inner_product(self.E[k], a)
                a -= self.E[k] * self.R[k, j]

            self.R[j, j] = self.VS.norm(a)

            # Modify E to take account of sign
            alpha = self.VS.inner_product(self.E[j], a)
            if abs(alpha) > 1e-17:
                self.E[j] *= - alpha / abs(alpha)

            # Vector defining next reflection
            self.V[j] = self.R[j, j] * self.E[j] - a

            # Orthonormalize V[j] with respect to orthonormal matrix E[0...j-1]
            self.V[j] -= self.VS.inner_product(self.E[:j], self.V[j]) @ self.E[:j]
            # Repeat orthonormalization. Mantra: "Twice is enough" [2]
            self.V[j] -= self.VS.inner_product(self.E[:j], self.V[j]) @ self.E[:j]
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
        _, sv, V_conj = np.linalg.svd(self.R)
        cond = (sv[1] - sv[-1]) / (sv[-2] - sv[-1])
        if cond > 1e+13:
            print('WARNING: Could not build surrogate in a stable way.')
            print('Relative range of singular values is {:.2e}.'.format(cond))
        q = V_conj[-1, :].conj()
        self.A_ring = RationalFunction(omegas, q, np.diag(q))

    def compute_surrogate(self, snapshots, omegas, greedy=True, tol=1e-2, n_iter=None):
        """Compute the rational surrogate"""
        if not greedy:
            self._build_surrogate(snapshots, omegas)
            return

        # Greedy: Take smallest and largest omegas as initial support points
        self.supports = [np.argmin(omegas), np.argmax(omegas)]
        is_eligible = np.ones(len(omegas), dtype=bool)
        is_eligible[self.supports] = False
        self._build_surrogate(snapshots[self.supports], omegas[self.supports])

        # Greedy: Add support points until relative surrogate error below tol
        if n_iter is None:
            n_iter = len(omegas)
        else:
            tol = 0.0
        t = 2
        while t < n_iter:
            reduced_argmin = self.A_ring.get_denominator_argmin(omegas[is_eligible])
            argmin = np.arange(len(omegas))[is_eligible][reduced_argmin]
            self.supports.append(argmin)
            is_eligible[argmin] = False
            A_hat = self.R @ self.A_ring(omegas[argmin])
            self._build_surrogate(snapshots[self.supports], omegas[self.supports], additive=True)
            rel_err = np.linalg.norm(self.R[:, -1] - np.append(A_hat, 0)) \
                    / np.linalg.norm(self.R[:, -1])
            if rel_err <= tol:
                break
            t += 1

    def get_surrogate(self, snapshots):
        """Returns the rational interpolation surrogate"""
        surrogate = self.A_ring
        surrogate.P = snapshots[self.supports].T * surrogate.q
        return surrogate

    def evaluate_surrogate(self, snapshots, z):
        """Evaluates the rational interpolation surrogate at points z"""
        return snapshots[self.supports].T @ self.A_ring(z)

    def get_interpolatory_eigenfrequencies(self, filtered=True, only_real=False):
        """Compute the eigenfrequencies as roots of rational interpolant"""
        eigfreqs = self.A_ring.roots(filtered)
        if only_real:
            return np.real(eigfreqs[np.isreal(eigfreqs)])
        return eigfreqs
