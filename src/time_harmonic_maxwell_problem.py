import numpy as np
import fenics as fen
import scipy.sparse

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath('__file__')))))
from src.rational_function import RationalFunction

class TimeHarmonicMaxwellProblem(object):
    """
    Finite element solver for solutions to the time harmonic
    Maxwell's equations formulated in the vector potential A
    
        \/ x ((1 / mu) \/ x A) - eps * omega^2 * A = j
    with boundary conditions
        A = A_D                     (Dirichlet boundaries, B_D)
        ((1 / mu) \/ x A) x n = g_N   (Neumann boundaries, B_N)
 
    Members
    -------
    V : dolfin.functions.functionspace.FunctionSpace
        Real FE space.
    A : list[dolfin.functions.function.TrialFunction]
        Trial function.
    v : dolfin.functions.function.TestFunction
        Test function.
    K : dolfin.cpp.la.Matrix
        Stiffness matrix.  
    M : dolfin.cpp.la.Matrix
        Mass matrix.
    L : dolfin.cpp.la.Vector
        Source term.
    N : dolfin.cpp.la.Vector
        Neumann boundary integral term.
    mu : dolfin.functions.expression.Expression
        Permeability.
    eps : dolfin.functions.expression.Expression
        Permittivity.
    j : dolfin.functions.expression.Expression
        Current density
    B_D : dolfin.cpp.mesh.SubDomain
        SubDomain object locating the Dirichlet boundary.
    B_N : dolfin.cpp.mesh.SubDomain
        SubDomain object locating the Neumann boundary.
    A_0 : dolfin.functions.expression.Expression
        Dirichlet boundary condition.
    g_N : dolfin.functions.expression.Expression
        Neumann boundary condition.
    A_sol : dolfin.functions.function.Function
        Solution to the variational problem.
    M_inner : dolfin.cpp.la.Matrix
        Matrix used to compute L2-norm in V.
    bc : dolfin.fem.bcs.DirichletBC
        Dirichlet boundary condition object.
    omega : list[float] or float
        Frequency for which the variational problem is solved.
    RI : RationalFunction
        The rational interpolant in barycentric coordinates.
    sigma : np.ndarray
        Singular values of the triangular matrix R.

    Methods
    -------
    setup() : None -> None
        Assemble the stiffness, mass, and source terms.
    solve(omega) : float -> None
        Computes the solution to the weak variational problem at omega.
    tosparse(A) : dolfin.cpp.la.Matrix -> scipy.sparse.csr_matrix
        Convert dolfin matrix to scipy sparse matrix in the CSR format.
    ...
    
    References
    ----------
    [1] FEniCS Project 2021: https://fenicsproject.org/
    [2] ...
    
    Usage
    -----
    Square waveguide with perfectly conducting walls and an inlet.
    >>> V = fen.FunctionSpace(fen.UnitSquareMesh(10, 10), 'P', 1)
    >>> mu = eps = fen.Constant(1.0)
    >>> j = fen.Constant(0.0)
    >>>
    >>> class B_D(fen.SubDomain):
    >>>     def inside(self, x, on_boundary):
    >>>         return not fen.near(x[0], 0.0)
    >>> class B_N(fen.SubDomain):
    >>>    def inside(self, x, on_boundary):
    >>>         return fen.near(x[0], 0.0)
    >>> 
    >>> A_D = fen.Constant(0.0)
    >>> g_N = fen.Constant(1.0)
    >>>
    >>> MP = MaxwellProblem(V, mu, eps, j, B_D(), B_N(), A_D, g_N)
    >>> MP.setup()
    >>> MP.solve(1)
    >>> A_sol = MP.get_solution()
    """

    def __init__(self, V, mu, eps, j, B_D, B_N, A_D, g_N):
        self.V = V
        self.K = None
        self.M = None
        self.L = None
        self.N = None
        self.mu = mu
        self.eps = eps
        self.j = j
        self.B_D = B_D
        self.B_N = B_N
        self.A_D = A_D
        self.g_N = g_N
        self.A_sol = None
        self.M_inner = None
        self.bc = None
        self.omega = None
        self.RI = None
        self.sigma = None

    def setup(self):
        """Assemble the stiffness and mass matrices with boundary conditions"""
        # Boundary function to identify Dirichlet and Neumann boundaries
        mesh = self.V.mesh()
        boundary_type = fen.MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
        boundary_type.set_all(0)
        self.B_D.mark(boundary_type, 1)
        self.B_N.mark(boundary_type, 2)

        A = fen.TrialFunction(self.V)
        v = fen.TestFunction(self.V)
        
        # Dirichlet boundary conditions
        self.bc = fen.DirichletBC(self.V, self.A_D, boundary_type, 1)

        # Neumann boundary conditions
        ds = fen.Measure('ds', subdomain_data=boundary_type)
        self.N = fen.assemble(fen.dot(self.g_N, v) * ds(2))

        # Assembly of stiffness, mass, and forcing term
        self.K = fen.assemble(1/self.mu * fen.dot(fen.curl(A), fen.curl(v)) * fen.dx)
        self.bc.apply(self.K)

        self.M = fen.assemble(self.eps * fen.dot(A, v) * fen.dx)
        self.bc.zero(self.M)

        self.L = fen.assemble(fen.dot(self.j, v) * fen.dx)
        self.bc.apply(self.L)

    def solve(self, omega, accumulate=False):
        """Solve the variational problem defined with .setup()"""
        if not accumulate:
            self.A_sol = []
            self.omega = []
        if isinstance(omega, (float, int)):
            omega = [omega]
        self.omega.extend(omega)
        for omg in omega:
            LHS = self.K - omg**2 * self.M
            RHS = self.L + self.N
            A = fen.Function(self.V)
            fen.solve(LHS, A.vector(), RHS)
            self.A_sol.append(A)

    def reset(self, **kwargs):
        """Clear all previous solutions and update attributes"""
        self.A_sol = []
        self.M_inner = None
        self.omega = []
        self.RI = None
        self.sigma = None
        self.__dict__.update(kwargs)
        self.setup()
            
    def restrict_solution_to_trace(self, trace):
        """Restrict the solution to a trace in the domain"""
        coords = self.V.tabulate_dof_coordinates()
        is_in_trace = [True if trace.inside(x, 'on_boundary') else False for x in coords]
        for i in range(len(self.A_sol)):
            A = self.A_sol[i].vector()
            #WG.A_sol[i] = fen.Function(WG.V)
            #WG.A_sol[i].vector()[:] = [a for j, a in enumerate(A) if trace.inside(coords[j], 'on_boundary')]
            self.A_sol[i] = [a for j, a in enumerate(A) if trace.inside(coords[j], 'on_boundary')]

    def restrict_solution_to_trace_integral(self, trace):
        mesh = self.V.mesh()
        boundary_type = fen.MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
        boundary_type.set_all(0)
        trace.mark(boundary_type, 1)
        ds = fen.Measure('ds', subdomain_data=boundary_type)
        L_norm = fen.assemble(fen.dot(fen.TrialFunction(self.V), fen.TestFunction(self.V)) * ds(1))
        for i in range(len(self.A_sol)):
            A = self.A_sol[i].vector()
            self.A_sol[i] = pow(((L_norm*A)*A).sum(), 0.5)
        
    @staticmethod
    def tosparse(A):
        """Convert dolfin matrix to a sparse SciPy matrix in CSR format"""
        A_mat = fen.as_backend_type(A).mat()
        A_sparse = scipy.sparse.csr_matrix(A_mat.getValuesCSR()[::-1])
        return A_sparse

    def get_V(self):
        """Return the finite element function space V"""
        return self.V

    def get_K(self, tosparse=True): 
        """Return the stiffness matrix K"""
        if tosparse:
            return self.tosparse(self.K)
        return self.K

    def get_M(self, tosparse=True):
        """Return the mass matrix M"""
        if tosparse:
            return self.tosparse(self.M)
        return self.M

    def get_L(self, tonumpy=True):
        """Return the source integral term L"""
        if tonumpy:
            return self.L.get_local()
        return self.L

    def get_N(self, tonumpy=True):
        """Return the Neumann boundary integral term N"""
        if tonumpy:
            return self.N.get_local()
        return self.N

    def get_solution(self, tonumpy=True):
        """Return the solution obtained with .solve()"""
        if tonumpy:
            return np.array([a.vector().get_local() for a in self.A_sol])
        return self.A_sol

    def get_boundary_indices_and_values(self):
        """Return list of indices and values of boundary points"""
        boundary_dict = self.bc.get_boundary_values()
        boundary_indices = list(boundary_dict.keys())
        boundary_values = list(boundary_dict.values())
        return boundary_indices, boundary_values

    def get_valid_indices(self):
        """Return list of indices that do not correspond to boundary points"""
        boundary_indices, _ = self.get_boundary_indices_and_values()
        all_indices = self.V.dofmap().dofs()
        valid_indices = np.delete(all_indices, boundary_indices)
        return valid_indices
    
    def insert_boundary_values(self, A_vec):
        """Insert boundary values into a vector with omitted boundary points"""
        boundary_indices, boundary_values = self.get_boundary_indices_and_values()
        valid_indices = self.get_valid_indices()
        A_vec_inserted = np.empty(self.V.dim())
        A_vec_inserted[valid_indices] = A_vec
        A_vec_inserted[boundary_indices] = boundary_values
        return A_vec_inserted

    def gram_schmidt(self, E, inner_product):
        """M-orthonormalize the columns of a matrix E"""
        E[0] /= pow(inner_product(E[0], E[0]), 0.5)
        for i in range(1, E.shape[0]):
            for j in range(i):
                E[i] -= inner_product(E[j], E[i]) * E[j]
            E[i] /= pow(inner_product(E[i], E[i]), 0.5)

    def get_orthonormal_matrix(self, shape, inner_product, seed=0):
        """Produce list of N orthonormal elements"""
        np.random.seed(seed)
        n1, n2 = shape
        E = np.random.randn(n1, n2)
        self.gram_schmidt(E, inner_product)
        return E

    def householder_triangularization(self, A, inner_product):
        """Compute the matrix R of a QR-decomposition of solutions at given frequencies"""
        N = A.shape[0]
        E = self.get_orthonormal_matrix(A.shape, inner_product)
        R = np.zeros((N, N))

        for k in range(N):
            R[k, k] = pow(inner_product(A[k], A[k]), 0.5)

            alpha = inner_product(E[k], A[k])
            if abs(alpha) > 1e-17:
                E[k] *= - alpha / abs(alpha)

            v = R[k, k] * E[k] - A[k]
            for j in range(k):
                v -= inner_product(E[j], v) * E[j]

            sigma = pow(inner_product(v, v), 0.5)
            if abs(sigma) > 1e-17:
                v /= sigma
            else:
                v = E[k]

            for j in range(k+1, N):
                A[j] -= 2 * v * inner_product(v, A[j])
                R[k, j] = inner_product(E[k], A[j])
                A[j] -= E[k] * R[k, j]
                
        return R

    def get_R(self):
        u = fen.TrialFunction(self.V)
        v = fen.TestFunction(self.V)
        M = self.tosparse(fen.assemble(fen.dot(u, v)*fen.dx))
        def inner_product(u, v):
            return (M.dot(u)).dot(v)
        A = self.get_solution(tonumpy=True)
        return self.householder_triangularization(A, inner_product)
    
    def compute_rational_interpolant(self):
        """Compute the rational interpolant based on the solution snapshots"""
        R = self.householder_triangularization()
        _, self.sigma, V = np.linalg.svd(R)
        q = V[-1, :].conj()
        P = self.get_solution(tonumpy=True).T * q
        self.RI = RationalFunction(self.omega, q, P)

    def get_interpolatory_eigenfrequencies(self, filtered=True):
        """Compute the eigenfrequencies based on the roots of the rational interpolant"""
        return self.RI.roots(filtered)
    

