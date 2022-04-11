import numpy as np
import fenics as fen
import scipy.linalg

class RationalFunction(object):
    """
    Encodes rational functions of the shape
    
                 sum_j P_ij / (z - z_j)
        r(z)_i = ----------------------
                 sum_j q_j / (z - z_j)
 
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

    References
    ----------
    [1] Klein G. Applications of Linear Barycentric Rational Interpolation
        https://core.ac.uk/download/pdf/20659062.pdf
    [2] ...
    
    Usage
    -----
    Define rational function r(z) = (1/z + 2/(z+1)) / (1/z + 1/(z+1))
    >>> RF = RationalFunction([0, 1], [1, 1], [1, 2])
    >>> RF.evaluate(1.)
            2.0
    >>> RF.roots()
            array([ 0.5])
    """
    def __init__(self, nodes, q, P):
        self.nodes = np.array(nodes)
        self.q = np.array(q)
        self.P = np.array(P)

    def __call__(self, z):
        return self.evaluate(z)

    def evaluate(self, z):
        zero_idx = np.where(np.abs(z - self.nodes) < 1e-10)[0]
        if len(zero_idx) > 0:
            return self.P.T[zero_idx[0]] / self.q[zero_idx[0]]
        A = self.P @ (z - self.nodes)**(-1)
        B = self.q @ (z - self.nodes)**(-1)
        return A / B

    def roots(self, filtered=True):
        N = len(self.nodes)
        A = np.diag(np.append(0, self.nodes))
        A[0, 1:] = self.q
        A[1:, 0] = 1
        B = np.diag(np.append(0, np.ones(N)))
        eigvals = scipy.linalg.eigvals(A, b=B)
        if filtered:
            eigvals[~np.isinf(eigvals)]
        return eigvals

    def get_numerator_min(self, samples):
        tiled_samples = np.tile(samples, (len(self.nodes), 1)).T
        B = self.q @ ((tiled_samples - self.nodes)**(-1)).T
        index_min = np.argmin(np.abs(B))
        return samples[index_min], index_min
    