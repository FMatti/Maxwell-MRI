# -*- coding: utf-8 -*-

import numpy as np
import scipy.linalg

class RationalFunction(object):
    """
    Encodes rational functions of the shape

                 P(z)_i   sum_j P_ij / (z - z_j)
        r(z)_i = ------ = ----------------------
                  q(z)    sum_j q_j / (z - z_j)

    Members
    -------
    nodes : list[float] or np.ndarray
        Sample points (z_j).
    q : list[float] or np.ndarray
        Numerator weights (q_j).
    P : list[float] or np.ndarray
        Denominator weights (P_ij).

    Methods
    -------
    evaluate(z) : list[float] or np.ndarray -> np.ndarray
        Evaluate rational function at a (list of) point(s) z.
    roots() : None -> float
        Return (filtered) roots of rational function. Filtered
        only returns roots between the smallest and largest node.
    get_nodes() : None -> np.ndarray
        Return the nodes z_j of the rational polynomial.
    get_denominator_argmin(samples) : list[float] -> float, float
        Return the sample (and its position) in a list of samples which
        minimizes the denominator polynomial q(z). 

    References
    ----------
    [1] Klein G. Applications of Linear Barycentric Rational Interpolation
        https://core.ac.uk/download/pdf/20659062.pdf

    Usage
    -----
    Define rational function r(z) = (1/z + 2/(z-1)) / (1/z + 1/(z-1))
    >>> RF = RationalFunction([0, 1], [1, 1], [1, 2])
    >>> RF.evaluate(1.0)
            array(2.0)
    >>> RF.roots()
            array([ 0.5 + 0.j])
    """
    def __init__(self, nodes, q, P):
        self.nodes = np.array(nodes, dtype=float)
        self.q = np.array(q)
        self.P = np.array(P)

    def __call__(self, z):
        return self.evaluate(z)

    def evaluate(self, z):
        """Evaluate the rational function at a point or list of points z."""
        C = np.subtract.outer(z, self.nodes, dtype=float)
        is_zero = np.isclose(C, 0.0, atol=1e-17)

        # Unproblematic case, where no differences in Cauchy matrix C are zero
        if not np.any(is_zero):
            return (self.P @ C.T**(-1)) / (self.q @ C.T**(-1))

        # Case where z is a single number
        if len(C.shape) == 1:
            return np.squeeze(self.P.T[is_zero].T / self.q[is_zero])
    
        has_zero = np.any(is_zero, axis=1)
        C[~has_zero] = C[~has_zero]**(-1)
        C[has_zero] = is_zero[has_zero]
        return (self.P @ C.T) / (self.q @ C.T)

    def roots(self, filtered=True):
        """Compute the roots of the denominator rational polynomial q(z)."""
        # Define eigenproblem as seen in [1]
        A = np.diag(np.append(0, self.nodes)).astype(complex)
        A[0, 1:] = self.q
        A[1:, 0] = 1
        B = np.diag(np.append(0, np.ones_like(self.nodes))).astype(complex)
        eigvals = scipy.linalg.eigvals(A, b=B)

        # Filter infinite eigenvalues
        if filtered:
            return eigvals[~np.isinf(eigvals)]
        return eigvals

    def get_nodes(self):
        """Return the nodes z_j used to build the rational function"""
        return self.nodes

    def get_denominator_argmin(self, samples):
        """Determine minimum denominator |q(z)| index in list of samples"""
        C = np.subtract.outer(samples, self.nodes, dtype=complex)
        has_zero = np.any(np.isclose(C, 0).T, axis=0).T
        B = self.q @ C[~has_zero].T**(-1)
        argmin_B = np.argmin(np.abs(B))
        return np.flatnonzero(~has_zero)[argmin_B]
