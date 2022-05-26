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
        if isinstance(self.g_N, list):
            self.N = [fen.assemble(fen.dot(g_N, v) * ds(2)) for g_N in self.g_N]
        else:
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

    def solve(self, omega, accumulate=False, solver='scipy'):
        """Solve the variational problem defined with .setup()"""
        if isinstance(omega, (float, int, complex)):
            omega = [omega]
        if isinstance(self.N, list):
            n = len(self.N)
        else:
            n = 1
        if not accumulate:
            k = 0
            self.solution = np.empty((len(omega)*n, self.V.dim()), dtype=complex)
            self.omega = np.repeat(omega, n)
        else:
            k = len(self.omega)
            self.solution = np.r_[self.solution, np.empty((len(omega)*n, self.V.dim()), dtype=complex)]
            self.omega = np.r_[self.omega, np.repeat(omega, n)]
        for omg in omega:
            if solver == 'scipy':
                LHS_re = self.get_K(tosparse=True) - omg**2 * self.get_M(tosparse=True)
                LHS_im = - 1j * omg * self.get_I(tosparse=True)
                RHS = self.get_N(tonumpy=True) + self.get_L(tonumpy=True)
                self.solution[k:k+n] = scipy.sparse.linalg.spsolve(LHS_re + LHS_im, RHS.T).T
                k += n
            elif solver == 'fenics':
                LHS = self.get_K(tosparse=False) - omg**2 * self.get_M(tosparse=False)
                if n > 1:
                    for i, N in enumerate(self.get_N(tonumpy=False)):
                        RHS = N + self.get_L(tonumpy=False)
                        u = fen.Function(self.V)
                        fen.solve(LHS, u.vector(), RHS)
                        self.solution[k+i] = u.vector().get_local()
                else:
                    RHS = self.get_N(tonumpy=False) + self.get_L(tonumpy=False)
                    u = fen.Function(self.V)
                    fen.solve(LHS, u.vector(), RHS)
                    self.solution[k] = u.vector().get_local()
                k += n

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
        I = self.get_I(tosparse=True)[inner_indices, :][:, inner_indices]
        if I.count_nonzero() > 0:
            n = len(inner_indices)
            identity = scipy.sparse.diags(np.ones(n), shape=(n, n), format='csr')
            empty = scipy.sparse.csr_matrix((n, n))
            K = scipy.sparse.vstack([scipy.sparse.hstack([empty, identity], format='csr'), scipy.sparse.hstack([K, -1j*I], format='csr')], format='csr')
            M = scipy.sparse.vstack([scipy.sparse.hstack([identity, empty], format='csr'), scipy.sparse.hstack([empty, M], format='csr')], format='csr') 
            eigvals, eigvecs = scipy.sparse.linalg.eigs(A=K, k=k, M=M, sigma=sigma, v0=v0)
        else:
            eigvals, eigvecs = scipy.sparse.linalg.eigsh(A=K, k=k, M=M, sigma=sigma, v0=v0)

        # Only return eigenfrequencies (square root of eigenvalues) in [a, b]
        if I.count_nonzero() == 0:
            eigvals = np.sqrt(eigvals)
        eigvals_in_ab = [e1 for e1 in eigvals if a <= e1 and e1 <= b]

        if len(eigvals_in_ab) == k:
            print(f'WARNING: Found exactly {k} eigenvalues within [{a}, {b}].')
            print('Increase parameter "k" to ensure all eigenvalues are found.')

        if return_eigvecs:
            if I.count_nonzero() == 0:
                eigvecs_in_ab = [e2 for e1, e2 in zip(eigvals, eigvecs.T) if a <= e1 and e1 <= b]
            else:
                eigvecs_in_ab = [e2[:n] for e1, e2 in zip(eigvals, eigvecs.T) if a <= e1 and e1 <= b]
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

    def get_I(self, tosparse=True):
        """Return the impedance boundary term I"""
        if tosparse:
            return self.tosparse(self.I)
        return self.I

    def get_L(self, tonumpy=True):
        """Return the source integral term L"""
        if tonumpy:
            return self.L.get_local()
        return self.L

    def get_N(self, tonumpy=True):
        """Return the Neumann boundary integral term N"""
        if tonumpy:
            if isinstance(self.N, list):
                return np.array([N.get_local() for N in self.N])
            return self.N.get_local()
        return self.N

    def get_solution(self, trace=None):
        """Return the solution obtained with .solve()"""
        if trace is not None:
            coords = self.V.tabulate_dof_coordinates()
            is_on_trace = lambda x: trace.inside(x, 'on_boundary')
            on_trace = np.apply_along_axis(is_on_trace, 1, coords)
            return self.solution[:, on_trace]
        return self.solution

    def save_solution(self, dirname=None, trace=None):
        SM = SnapshotMatrix(self.get_solution(trace=trace), self.omega)
        if dirname is None:
            return SM
        with open(dirname, 'wb') as file:
            pickle.dump(SM, file)

    def load_solution(self, dirname):
        with open(dirname, 'rb') as file:
            SM = pickle.load(file)
        self.solution = SM.get_solution()
        self.omega = SM.get_frequency()

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
