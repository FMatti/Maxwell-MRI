# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import pickle

import fenics as fen

from .snapshot_matrix import SnapshotMatrix

class TimeHarmonicMaxwellProblem(object):
    """
    Finite element solver for solutions to the time harmonic
    Maxwell's equations formulated in the vector potential A
    
        \/ x ((1 / mu) \/ x u) - eps * omega^2 * u = j
    with boundary conditions

        u = u_D
            (Dirichlet boundaries, B_D)

        ((1 / mu) \/ x u) x n = g_N
            (Neumann boundaries, B_N)

        n x ((1 / mu) \/ x u) - i * omega * imp * (n x u) x n = 0
            (Impedance boundaries, B_I [3])
 
    Members
    -------
    V : dolfin.functions.functionspace.FunctionSpace
        Real FE space.
    u : list[dolfin.functions.function.TrialFunction]
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
    I : dolfin.cpp.la.Vector
        Impedance boundary integral term.
    mu : dolfin.functions.expression.Expression
        Permeability.
    eps : dolfin.functions.expression.Expression
        Permittivity.
    j : dolfin.functions.expression.Expression
        Current density
    B_D : dolfin.cpp.mesh.SubDomain
        SubDomain object locating the Dirichlet boundary.
    u_D : dolfin.functions.expression.Expression
        Dirichlet boundary condition.
    B_N : dolfin.cpp.mesh.SubDomain, optional
        SubDomain object locating the Neumann boundary.
    g_N : dolfin.functions.expression.Expression, optional
        Neumann boundary condition.
    B_I : dolfin.cpp.mesh.SubDomain, optional
        SubDomain object locating the impedance boundary.
    imp : dolfin.functions.expression.Expression, optional
        Impedance.
    solution : dolfin.functions.function.Function
        Solution to the variational problem.
    omega : list[float] or float
        Frequency for which the variational problem is solved.
    bc : dolfin.fem.bcs.DirichletBC
        Dirichlet boundary condition object.

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
        Model Order Reduction based on minimal rational approximation.
    [3] Monk P. Finite Element Methods for Maxwell's Equations
        Oxford University Press, 2003.

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
    >>> u_D = fen.Constant(0.0)
    >>> g_N = fen.Constant(1.0)
    >>>
    >>> MP = TimeHarmonicMaxwellProblem(V, mu, eps, j, B_D(), u_D, B_N(), g_N)
    >>> MP.setup()
    >>> MP.solve(1)
    >>> solution = MP.get_solution()
    """

    def __init__(self, V, mu, eps, j, B_D, u_D, B_N=None, g_N=None, B_I=None, imp=None):
        self.V = V
        self.K = None
        self.M = None
        self.L = None
        self.N = None
        self.I = None
        self.mu = mu
        self.eps = eps
        self.j = j
        self.B_D = B_D
        self.u_D = u_D
        self.B_N = B_N
        self.g_N = g_N
        self.B_I = B_I
        self.imp = imp
        self.solution = None
        self.omega = None
        self.bc = None

        class EmptyBoundary(fen.SubDomain):
                def inside(self, x, on_boundary):
                    return False

        if B_N is None:
            self.B_N = EmptyBoundary()

        if g_N is None:
            if self.V.mesh().topology().dim() == 2:
                self.g_N = fen.Expression('0.0', degree=2)
            else:
                self.g_N = fen.Expression(('0.0', '0.0', '0.0'), degree=2)

        if B_I is None:
            self.B_I = EmptyBoundary()

        if imp is None:
            self.imp = fen.Expression('0.0', degree=2)

    def setup(self):
        """Assemble the stiffness and mass matrices with boundary conditions"""
        # Boundary function to identify Dirichlet and Neumann boundaries
        mesh = self.V.mesh()
        boundary_type = fen.MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
        boundary_type.set_all(0)
        self.B_D.mark(boundary_type, 1)
        self.B_N.mark(boundary_type, 2)
        self.B_I.mark(boundary_type, 3)

        u = fen.TrialFunction(self.V)
        v = fen.TestFunction(self.V)

        ds = fen.Measure('ds', subdomain_data=boundary_type)
        n = fen.FacetNormal(mesh)
 
        # Dirichlet boundary conditions
        self.bc = fen.DirichletBC(self.V, self.u_D, boundary_type, 1)

        # Neumann boundary conditions
        self.N = fen.assemble(fen.dot(self.g_N, v) * ds(2))

        # Impedance boundary condition
        if mesh.topology().dim() == 2:
            self.I = fen.assemble(self.imp * fen.dot(u, v) * ds(3))
        else:
            self.I = fen.assemble(self.imp * fen.dot(fen.cross(fen.cross(n, u), n), v) * ds(3))

        # Assembly of stiffness, mass, and forcing term
        self.K = fen.assemble(1/self.mu * fen.dot(fen.curl(u), fen.curl(v)) * fen.dx)
        self.bc.apply(self.K)

        self.M = fen.assemble(self.eps * fen.dot(u, v) * fen.dx)
        self.bc.zero(self.M)

        self.L = fen.assemble(fen.dot(self.j, v) * fen.dx)
        self.bc.apply(self.L)

    def solve(self, omega, accumulate=False):
        """Solve the variational problem defined with .setup()"""
        if not accumulate:
            self.solution = []
            self.omega = []
        if isinstance(omega, (float, int)):
            omega = [omega]
        self.omega.extend(omega)
        for omg in omega:
            LHS = self.K - omg**2 * self.M
            RHS = self.L + self.N
            u = fen.Function(self.V)
            fen.solve(LHS, u.vector(), RHS)
            self.solution.append(u)

    def complex_solve(self, omega, accumulate=False):
        """Solve the variational problem defined with .setup()"""
        if isinstance(omega, (float, int)):
            omega = [omega]
        if not accumulate:
            self.solution = np.empty((len(omega), self.V.dim()*2), dtype=float)
            self.omega = []
        self.omega.extend(omega)
        for i, omg in enumerate(omega):
            LHS_re = self.tosparse(self.K) - omg**2 * self.tosparse(self.M)
            LHS_im = - omg * self.tosparse(self.I)
            RHS_re = self.L.get_local() + self.N.get_local()
            RHS_im = np.zeros_like(RHS_re)
            LHS = scipy.sparse.vstack([scipy.sparse.hstack([LHS_re, -LHS_im], format='csr'),
                                       scipy.sparse.hstack([LHS_im, LHS_re], format='csr')], format='csr')
            RHS = np.r_[RHS_re, RHS_im]
            u = scipy.sparse.linalg.spsolve(LHS, RHS)
            if not accumulate:
                self.solution[i] = u
            else:
                # This is bullshit (port all to numpy later on please)
                self.solution.append(u)

    def get_numerical_eigenfrequencies(self, a=-np.inf, b=np.inf, k=10, v0=None, return_eigvecs=False):
        """Solve an eigenvalue problem K*v = omega^2*M*v"""
        if a == -np.inf or b == np.inf:
            sigma = None
        else:
            sigma = (a + b) / 2

        # Only use non-zero (not on boundary) components in M and K
        inner_indices = self.get_inner_indices()
        if v0 is not None:
            v0 = v0[inner_indices]
        K = self.get_K(tosparse=True)[inner_indices, :][:, inner_indices]
        M = self.get_M(tosparse=True)[inner_indices, :][:, inner_indices]
        eigvals, eigvecs = scipy.sparse.linalg.eigsh(A=K, k=k, M=M, sigma=sigma, v0=v0)

        # Only return eigenfrequencies (square root of eigenvalues) in [a, b]
        eigvals = np.sqrt(eigvals)
        eigvals_in_ab = [e1 for e1 in eigvals if a <= e1 and e1 <= b]

        if len(eigvals_in_ab) == k:
            print(f'WARNING: Found exactly {k} eigenvalues within [{a}, {b}].')
            print('Increase parameter "k" to ensure all eigenvalues are found.')

        if return_eigvecs:
            eigvecs_in_ab = [e2 for e1, e2 in zip(eigvals, eigvecs.T) if a <= e1 and e1 <= b]
            return eigvals_in_ab, eigvecs_in_ab
        return eigvals_in_ab

    @staticmethod
    def tosparse(A):
        """Convert dolfin matrix to a sparse SciPy matrix in CSR format"""
        A_mat = fen.as_backend_type(A).mat()
        A_shape = (A.size(0), A.size(1))
        A_sparse = scipy.sparse.csr_matrix(A_mat.getValuesCSR()[::-1], shape=A_shape)
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

    def get_I(self, tonumpy=True):
        """Return the Neumann boundary integral term N"""
        if tonumpy:
            return self.I.get_local()
        return self.I

    def get_solution(self, tonumpy=True, trace=None):
        """Return the solution obtained with .solve()"""
        if trace is not None:
            coords = self.V.tabulate_dof_coordinates()
            is_on_trace = lambda x: trace.inside(x, 'on_boundary')
            on_trace = np.apply_along_axis(is_on_trace, 1, coords)
            if isinstance(self.solution, np.ndarray):
                return self.solution[:, on_trace]
            return np.array([a.vector().get_local()[on_trace] for a in self.solution])
        if tonumpy:
            if isinstance(self.solution, np.ndarray):
                # Bullshit (port all to numpy later)
                return self.solution[:, self.solution.shape[1] // 2:  ]
            return np.array([a.vector().get_local() for a in self.solution])
        return self.solution

    def save_solution(self, dirname, trace=None):
        SM = SnapshotMatrix(self.get_solution(tonumpy=True, trace=trace), self.omega)
        with open(dirname, 'wb') as file:
            pickle.dump(SM, file)

    def load_solution(self, dirname):
        with open(dirname, 'rb') as file:
            SM = pickle.load(file)
        self.solution = SM.get_snapshots()
        self.omega = SM.get_frequencies()

    def get_frequency(self):
        """Return the frequencies corresponding to the solutions"""
        return np.array(self.omega)

    def get_boundary_indices_and_values(self):
        """Return list of indices and values of boundary points"""
        boundary_dict = self.bc.get_boundary_values()
        return list(boundary_dict.keys()), list(boundary_dict.values())

    def get_inner_indices(self):
        """Return list of indices that do not correspond to boundary points"""
        boundary_indices, _ = self.get_boundary_indices_and_values()
        all_indices = self.V.dofmap().dofs()
        return np.delete(all_indices, boundary_indices)
    
    def insert_boundary_values(self, u_vec):
        """Insert boundary values into a vector with omitted boundary points"""
        boundary_indices, boundary_values = self.get_boundary_indices_and_values()
        inner_indices = self.get_inner_indices()
        u_vec_inserted = np.empty(self.V.dim())
        u_vec_inserted[inner_indices] = u_vec
        u_vec_inserted[boundary_indices] = boundary_values
        return u_vec_inserted
