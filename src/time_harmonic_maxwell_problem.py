import numpy as np
import fenics as fen
import scipy.sparse

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
    A : dolfin.functions.function.TrialFunction
        Trial function.
    v : dolfin.functions.function.TestFunction
        Test function.
    K : dolfin.cpp.la.Matrix
        Stiffness matrix.  
    M : dolfin.cpp.la.Matrix
        Mass matrix.
    L : dolfin.cpp.la.Vector
        Source term.
    L : dolfin.cpp.la.Vector
        Neumann boundary term.
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
    g : dolfin.functions.expression.Expression
        Neumann boundary condition.
    A_sol : dolfin.functions.function.Function
        Solution to the variational problem.
    F_norm : dolfin.cpp.la.Matrix
        Matrix used to compute L2-norm of A_sol.
    bc : dolfin.fem.bcs.DirichletBC
        Dirichlet boundary condition object.
    omega : float
        Frequency for which the variational problem is solved.

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
        self.A = fen.TrialFunction(self.V)
        self.v = fen.TestFunction(self.V)
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
        self.F_norm = None
        self.bc = None
        self.omega = None

    def setup(self):
        """Assemble the stiffness and mass matrices with boundary conditions"""
        # Boundary function to identify Dirichlet and Neumann boundaries
        mesh = self.V.mesh()
        boundary_type = fen.MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
        boundary_type.set_all(0)
        self.B_D.mark(boundary_type, 1)
        self.B_N.mark(boundary_type, 2)

        # Dirichlet boundary conditions
        self.bc = fen.DirichletBC(self.V, self.A_D, boundary_type, 1)

        # Neumann boundary conditions
        ds = fen.Measure('ds', subdomain_data=boundary_type)
        self.N = fen.assemble(fen.dot(self.g_N, self.v) * ds(2))

        # Assembly of stiffness, mass, and forcing term
        self.K = fen.assemble(1/self.mu * fen.dot(fen.curl(self.A), fen.curl(self.v)) * fen.dx)
        self.bc.apply(self.K)

        self.M = fen.assemble(self.eps * fen.dot(self.A, self.v) * fen.dx)
        self.bc.zero(self.M)

        self.L = fen.assemble(fen.dot(self.j, self.v) * fen.dx)
        self.bc.apply(self.L)

    def solve(self, omega):
        """Solve the variational problem defined with .setup()"""
        self.omega = omega
        LHS = self.K - omega**2 * self.M
        RHS = self.L + self.N
        self.A_sol = fen.Function(self.V)
        fen.solve(LHS, self.A_sol.vector(), RHS)

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
        else:
            return self.K
    
    def get_M(self, tosparse=True):
        """Return the mass matrix M"""
        if tosparse:
            return self.tosparse(self.M)
        else:
            return self.M
    
    def get_L(self, get_local=True):
        """Return the source integral term L"""
        if get_local:
            return self.L.get_local()
        else:
            return self.L

    def get_N(self, get_local=True):
        """Return the Neumann boundary integral term N"""
        if get_local:
            return self.N.get_local()
        else:
            return self.N
     
    def get_solution(self):
        """Return the solution obtained with .solve()"""
        return self.A_sol

    def compute_solution_norm(self):
        """Compute the L2-norm of the solution obtained with .solve()"""
        if self.F_norm is None:
            self.F_norm = fen.assemble(fen.dot(fen.TrialFunction(self.V), fen.TestFunction(self.V)) * fen.dx)
        return pow((self.A_sol.vector()*(self.F_norm*self.A_sol.vector())).sum(), 0.5)
    
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