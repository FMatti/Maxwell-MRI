import numpy as np
import fenics as fen
import scipy.sparse

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath('__file__')))))
from src.rational_function import RationalFunction
import src.helpers as helpers

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
    sv : np.ndarray
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
    [2] Pradovera D. and Nobile F. Frequency-domain non-intrusive greedy
        Model Order Reduction based on minimal rational approximation
    [3] ...
    
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
        self.sv = None

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

    def get_solution(self, tonumpy=True, trace=None):
        """Return the solution obtained with .solve()"""
        if trace is not None:
            coords = self.V.tabulate_dof_coordinates()
            is_on_trace = lambda x: trace.inside(x, 'on_boundary')
            on_trace = np.apply_along_axis(is_on_trace, 1, coords)
            return np.array([a.vector().get_local()[on_trace] for a in self.A_sol])
        if tonumpy:
            return np.array([a.vector().get_local() for a in self.A_sol])
        return self.A_sol

    def get_frequency(self):
        """Return the frequencies corresponding to the solutions"""
        return self.omega

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

    def compute_surrogate(self, VS, additive=False, R=None, E=None, V=None):
        """Compute the rational surrogate with previously computed snapshots"""
        A = self.get_solution(tonumpy=True, trace=VS.get_trace())
        if additive:
            R, E, V = helpers.householder_triangularization(A, VS, R, E, V, returns=True)
        else:
            R = helpers.householder_triangularization(A, VS)
        _, self.sv, V_ = np.linalg.svd(R)
        q = V_[-1, :].conj()
        P = A.T * q
        omega = self.get_frequency()
        self.RI = RationalFunction(omega, q, P)
        if additive:
            return R, E, V

    def compute_greedy_surrogate(self, VS, a, b, tol=1e-2, n=1000):
        """Compute the rational surrogate with the greedy algorithm"""
        self.solve([a, b])
        R, E, V = self.compute_surrogate(VS, additive=True)
        samples = np.linspace(a, b, n)[1:-1]
        while len(samples) > 0:
            samples_min, index_min = self.RI.get_numerator_min(samples)
            self.solve(samples_min, accumulate=True)
            a = self.get_solution(tonumpy=True)[-1]
            if VS.norm(a - self.RI(samples_min)) <= tol*VS.norm(a):
                # Compute surrogate using the last snapshot before termination
                self.compute_surrogate(VS, additive=True, R=R, E=E, V=V)
                break
            samples = np.delete(samples, index_min)
            R, E, V = self.compute_surrogate(VS, additive=True, R=R, E=E, V=V)
            # if [surrogate could not be built in stable way, maybe np.isclose(WG.sv[-2:], WG.sv[-1:])]
            #    go back to last surrogate?!
            
    @staticmethod
    def get_numerator_argmin(RF, choices):
        tiled_choices = np.tile(choices, (len(RF.nodes), 1)).T
        index_min = np.argmin(np.abs(RF.q @ (tiled_choices - RF.nodes).T**(-1)))
        return index_min

    def compute_greedy_surrogate(self, VS, a, b, tol=1e-2):
        """Compute the rational surrogate in a greedy manner [2]"""
        omegas = [a, b]
        self.solve(omegas)
        self.compute_surrogate(VS, omegas)
        choices = np.linspace(a, b, 1000)[1:-1]
        while len(choices) > 0:
            index_min = self.get_numerator_argmin(self.RI, choices)
            omega_min = choices[index_min]
            self.solve(omega_min, accumulate=True)
            A = self.get_solution(tonumpy=True)[-1]
            if VS.norm(A - self.RI(omega_min)) <= tol*VS.norm(A):
                break
            choices = np.delete(choices, index_min)
            omegas.append(omega_min)
            self.compute_surrogate(VS, omegas)

    def get_interpolatory_eigenfrequencies(self, filtered=True):
        """Compute the eigenfrequencies based on the roots of the rational interpolant"""
        return self.RI.roots(filtered)

    def get_error(self, VS):
        N = len(self.omega)
        A = self.get_solution(tonumpy=True, trace=VS.get_trace())
        relative_error = np.empty(N)
        FE_norm = np.empty(N)
        RI_norm = np.empty(N)
        for i in range(N):
            FE_norm[i] = VS.norm(A[i])
            RI_norm[i] = VS.norm(self.RI(self.omega[i]))
            relative_error[i] = VS.norm(A[i] - self.RI(self.omega[i])) / FE_norm[i]
        return relative_error, FE_norm, RI_norm
    