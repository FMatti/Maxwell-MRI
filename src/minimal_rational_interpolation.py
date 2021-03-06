# -*- coding: utf-8 -*-

import numpy as np
import copy

from .rational_function import RationalFunction

class MinimalRationalInterpolation(object):
    """
    Minimal rational interpolation.

    Members
    -------
    VS : VectorSpace
        Vector space object.
    u_ring : RationalFunction
        Constituent of the rational interpolant in barycentric coordinates.
    R : None or np.ndarray
        Upper triangular matrix (N_R x N_R) obtained from the Householder
        triangularization of the first N_R columns in u.
    E : np.ndarray
        Orthonormal matrix created in Householder triangularization.
    V : np.ndarray
        Householder matrix created in Householder triangularization.
    supports : list
        Indices of support points used to build surrogate from snapshots.    

    Methods
    -------
    compute_surrogate(target, ...) : TimeHarmonicMaxwellProblem, ... -> None
        Compute the rational interpolation surrogate for a THMP.
    get_surrogate(snapshots, z) : np.ndarray, np.ndarray -> RationalFunction
        Returns the computed rational interpolation surrogate.
    get_interpolatory_eigenfrequencies() : None -> np.ndarray
        Returns the eigenfrequencies determined with the surrogate.
    evaluate_surrogate(snapshots, z) : np.ndarray, np.ndarray -> np.ndarray 
        Evaluates the rational surrogate at point or list of points z.

    References
    ----------
    [1] Pradovera D. and Nobile F.: Frequency-domain non-intrusive greedy
        Model Order Reduction based on minimal rational approximation
        Springer International Publ. (2021). DOI: 10.1007/978-3-030-84238-3_16
    [2] Giraud L. and Langou J. and Rozloznk M.: On the round-off error
        analysis of the Gram-Schmidt algorithm with reorthogonalization.
        URL: https://www.cerfacs.fr/algor/reports/2002/TR_PA_02_33.pdf
    [3] Lloyd N. Trefethen: Householder triangularization of a quasimatrix.
        IMA Journal of Numerical Analysis (2008). DOI: 10.1093/imanum/dri017
    [4] Bonzzoni F., Pradovera D., and Ruggeri M.: Rational-based model order
        reduction of Helmholtz frequency response problems with adaptive finite
        element snapshots. (2021). DOI: 10.48550/arXiv.2112.04302
    """
    def __init__(self, VS):
        self.VS = VS
        self.u_ring = None
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
            self.E = np.array(np.random.randn(n1, n2), dtype=complex)
        else:
            # Extend orthonormal matrix E to orthonormal matrix with given shape
            n1 -= self.E.shape[0]
            self.E = np.r_[self.E, np.random.randn(n1, n2)]
        self._gram_schmidt(self.E, n1)

    def _householder_triangularization(self, u):
        """(Sequentially) compute the upper triangular matrix of a QR-decomposi-
        tion of the snapshot matrix A of a time-harmonic Maxwell problem."""

        N_u = u.shape[0]

        # Declare matrix R or extend it with zeros if it already exists
        if self.R is None:
            N_R = 0
            self.R = np.zeros((N_u, N_u), dtype=complex)
        else:
            N_R = self.R.shape[0]
            self.R = np.pad(self.R, (0, N_u-N_R), mode='constant')

        # Get/extend orthonormal matrix E to the shape of snapshot matrix
        self._set_orthonormal_matrix(u.shape)

        # Declare/extend matrix V for Householder vectors
        if self.V is None:
            self.V = np.empty((N_u, u.shape[1]), dtype=complex)
        else:
            self.V = np.pad(self.V, ((0, N_u-N_R), (0, 0)), mode='constant')

        for j in range(N_R, N_u):
            a = u[j].copy()

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

        # Check conditioning of build
        cond = (sv[1] - sv[-1]) / (sv[-2] - sv[-1])
        if cond > 1e+13:
            print('WARNING: Could not build surrogate in a stable way.')
            print('Relative range of singular values is {:.2e}.'.format(cond))
        
        # Rational surrogate as defined in MRI [4] in reduced-order form u_ring
        q = V_conj[-1, :].conj()
        self.u_ring = RationalFunction(omegas, q, np.diag(q))

    def compute_surrogate(self, target, omegas=None, greedy=True, tol=1e-2,
                          n_iter=None, return_history=False, return_relerr=False,
                          solver='scipy'):
        """Compute the rational surrogate"""
        # In the non-greedy mode, use all omegas to build rational surrogate
        if not greedy:
            self.supports = target.get_frequency()
            self._build_surrogate(target.get_solution(trace=self.VS.trace), self.supports)
            return

        # Take smallest and largest omegas as initial support points
        self.supports = [np.argmin(omegas), np.argmax(omegas)]
        is_eligible = np.ones(len(omegas), dtype=bool)
        is_eligible[self.supports] = False
        target.solve(omegas[self.supports], solver=solver)
        self._build_surrogate(target.get_solution(trace=self.VS.trace), omegas[self.supports])

        # Distinguish the case where a fixed number of iterations are desired
        if n_iter is None:
            n_iter = len(omegas)
        else:
            tol = 0.0

        # Keep track of surrogate history or relative error history
        if return_history:
            surrogate_history = [self.get_surrogate(target.get_solution(trace=self.VS.trace))]
        if return_relerr:
            relerr_history = []

        # Add support points until relative surrogate error below tol
        t = 2
        while t < n_iter:
            reduced_argmin = self.u_ring.get_denominator_argmin(omegas[is_eligible])
            argmin = np.arange(len(omegas))[is_eligible][reduced_argmin]
            self.supports.append(argmin)
            is_eligible[argmin] = False
            u_hat = self.R @ self.u_ring(omegas[argmin])
            target.solve(omegas[argmin], accumulate=True, solver=solver)
            self._build_surrogate(target.get_solution(trace=self.VS.trace), omegas[self.supports], additive=True)
            if return_history:
                surrogate_history.append(self.get_surrogate(target.get_solution(trace=self.VS.trace)))
            
            # Approximate relative error with euclidean norms
            rel_err = np.linalg.norm(self.R[:, -1] - np.append(u_hat, 0)) \
                    / np.linalg.norm(self.R[:, -1])
            if return_relerr:
                relerr_history.append(rel_err)
            if rel_err <= tol:
                break
            t += 1

        if return_history and return_relerr:
            return surrogate_history, relerr_history
        elif return_history:
            return surrogate_history
        elif return_relerr:
            return relerr_history

    def get_surrogate(self, snapshots):
        """Returns the rational interpolation surrogate"""
        surrogate = copy.copy(self.u_ring)
        if snapshots.shape[0] == len(self.supports):
            surrogate.P = snapshots.T * surrogate.q
        else:
            surrogate.P = snapshots[self.supports].T * surrogate.q
        return surrogate

    def evaluate_surrogate(self, snapshots, z):
        """Evaluates the rational interpolation surrogate at points z"""
        if snapshots.shape[0] == len(self.supports):
            return snapshots.T @ self.u_ring(z)
        else:
            return snapshots[self.supports].T @ self.u_ring(z)

    def get_interpolatory_eigenfrequencies(self, filtered=True, only_real=False):
        """Compute the eigenfrequencies as roots of rational interpolant"""
        eigfreqs = self.u_ring.roots(filtered)
        if only_real:
            return np.real(eigfreqs[np.isreal(eigfreqs)])
        return eigfreqs
