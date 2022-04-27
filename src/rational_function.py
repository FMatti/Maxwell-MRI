# -*- coding: utf-8 -*-

from termios import B1152000
import numpy as np
import scipy.linalg

class RationalFunction(object):
    """
    Encodes rational functions of the shape

                 P(z)   sum_j P_ij / (z - z_j)
        r(z)_i = ---- = ----------------------
                 q(z)   sum_j q_j / (z - z_j)

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
    evaluate(z) : float -> float
        Evaluate rational function at a point z.
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
    >>> RF.evaluate(1.)
            2.0
    >>> RF.roots()
            array([ 0.5])
    """
    def __init__(self, nodes, q, P):
        self.nodes = np.array(nodes, dtype=float)
        self.q = np.array(q)
        self.P = np.array(P)

    def __call__(self, z):
        return self.evaluate(z)

    def evaluate(self, z):
        diff = np.subtract.outer(self.nodes, z)
        zero_diff = np.isclose(diff, 0)
        if not np.any(zero_diff):
            return (self.P @ diff**(-1)) / (self.q @ diff**(-1))
        if len(diff.shape) == 1:
            return self.P.T[zero_diff] / self.q[zero_diff]
        is_pole = np.any(zero_diff, axis=0)
        dim = 1 if len(self.P.shape) == 1 else self.P.shape[0]
        values = np.empty((dim, diff.shape[1]))
        A = self.P @ diff[:, ~is_pole]**(-1)
        B = self.q @ diff[:, ~is_pole]**(-1)
        values[:, ~is_pole] = A / B
        for pole in np.flatnonzero(is_pole):
            A = self.P.T[zero_diff[:, pole]].flatten()
            B = self.q[zero_diff[:, pole]]
            values[:, pole] = A / B
        return np.squeeze(values)

    def roots(self, filtered=True):
        N = len(self.nodes)
        A = np.diag(np.append(0, self.nodes))
        A[0, 1:] = self.q
        A[1:, 0] = 1
        B = np.diag(np.append(0, np.ones(N)))
        eigvals = scipy.linalg.eigvals(A, b=B)
        if filtered:
            return eigvals[~np.isinf(eigvals)]
        return eigvals

    def get_nodes(self):
        return self.nodes

    def get_denominator_argmin(self, samples):
        subtractions = np.subtract.outer(self.nodes, samples)
        has_zero = np.any(np.isclose(subtractions, 0), axis=0)
        B = subtractions.T[~has_zero]**(-1) @ self.q
        argmin_B = np.argmin(np.abs(B))
        return np.flatnonzero(~has_zero)[argmin_B]
