import numpy as np
import fenics as fen
import scipy.linalg

class VectorSpace(object):
    """
    Abstract class encoding an inner product space.

    Members
    -------
    matrix : scipy.sparse.coo_matrix
        Matrix representation of the inner product.
    trace : fen.SubDomain
        Subdomain locating the trace.

    Methods
    -------
    inner_product(u, v) : np.ndarray -> float
        Inner product between two vectors in the space.
    norm(u) : np.ndarray -> float
        Norm of a vector in the space.
    get_trace() : None -> fen.SubDomain
        Returns the subdomain object locating the trace.
    """
    def __init__(self):
        self.matrix = None
        self.trace = None

    def inner_product(self, u, v):
        return (self.matrix.dot(u)).dot(v)

    def norm(self, u):
        return pow(self.inner_product(u, u), 0.5)

    def get_trace(self):
        return self.trace

class VectorSpaceL2(VectorSpace):
    """
    L2 vector space for a time-harmonic Maxwell problem, either
    involving the whole space or just a subspace defined by a
    trace.
    """
    def __init__(self, THMP, trace=None):
        self.trace = trace
        u = fen.TrialFunction(THMP.V)
        v = fen.TestFunction(THMP.V)
        if trace is None:
            form = fen.assemble(fen.dot(u, v)* fen.dx)
            self.matrix = THMP.tosparse(form)
        else:
            mesh = THMP.get_V().mesh()
            indicator = fen.MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
            indicator.set_all(0)
            self.trace.mark(indicator, 1)
            ds = fen.Measure('ds', subdomain_data=indicator)
            form = fen.assemble(fen.dot(u, v)* ds(1))
            coords = THMP.V.tabulate_dof_coordinates()
            is_on_trace = lambda x: self.trace.inside(x, 'on_boundary')
            on_trace = np.apply_along_axis(is_on_trace, 1, coords)
            self.matrix = THMP.tosparse(form)[on_trace, :][:, on_trace]
