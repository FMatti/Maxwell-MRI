# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse

import fenics as fen

from .rational_function import RationalFunction

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
        #self.RI = None
        #self.sv = None

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
"""
    def compute_surrogate(self, VS, additive=False, R=None, E=None, V=None):
        #Compute the rational surrogate with previously computed snapshots
        A = self.get_solution(tonumpy=True, trace=VS.get_trace())
        if additive:
            R, E, V = self.householder_triangularization(A, VS, R, E, V, returns=True)
        else:
            R = self.householder_triangularization(A, VS)
        _, self.sv, V_ = np.linalg.svd(R)
        # Check stability
        q = V_[-1, :].conj() 
        P = A.T * q
        omega = self.get_frequency()
        self.RI = RationalFunction(omega, q, P)
        if additive:
            return R, E, V

    def compute_greedy_surrogate(self, VS, a, b, tol=1e-2, n=1000):
        #Compute the rational surrogate with the greedy algorithm
        self.solve([a, b])
        R, E, V = self.compute_surrogate(VS, additive=True)
        samples = np.linspace(a, b, n)[1:-1]
        while len(samples) > 0:
            samples_min, index_min = self.RI.get_denominator_argmin(samples)
            self.solve(samples_min, accumulate=True)
            a = self.get_solution(tonumpy=True, trace=VS.get_trace())[-1]
            if VS.norm(a - self.RI(samples_min)) <= tol*VS.norm(a):
                # Compute surrogate using the last snapshot before termination
                self.compute_surrogate(VS, additive=True, R=R, E=E, V=V)
                break
            samples = np.delete(samples, index_min)
            R, E, V = self.compute_surrogate(VS, additive=True, R=R, E=E, V=V)
            # if [surrogate could not be built in stable way, maybe np.isclose(WG.sv[-2:], WG.sv[-1:])]
            #    go back to last surrogate?!

    def get_interpolatory_eigenfrequencies(self, filtered=True):
        #Compute the eigenfrequencies based on the roots of the rational interpolant
        return self.RI.roots(filtered)

    def get_error(self, VS):
        N = len(self.omega)
        A = self.get_solution(tonumpy=True, trace=VS.get_trace())
        relative_error = np.empty(N)
        FE_norm = np.empty(N)
        RI_norm = np.empty(N)
        for i in range(N):
            FE_norm[i] = VS.norm(A[i])
            relative_error[i] = VS.norm(A[i] - self.RI(self.omega[i])) / FE_norm[i]
        return relative_error


    def gram_schmidt(self, E, VS, k=None):
        #M-orthonormalize the (k last) rows of a matrix E
        if k is None or k == E.shape[0]:
            k = E.shape[0]
            E[0] /= VS.norm(E[0])
        for i in range(E.shape[0]-k, E.shape[0]):
            for j in range(i):
                E[i] -= VS.inner_product(E[j], E[i]) * E[j]
            E[i] /= VS.norm(E[i])
            # Twice is enough
            for j in range(i):
                E[i] -= VS.inner_product(E[j], E[i]) * E[j]
            E[i] /= VS.norm(E[i])

    def get_orthonormal_matrix(self, shape, VS, E=None):
        #Produce (extension of) orthonormal matrix with given shape
        n1, n2 = shape
        if E is None:
            E_on = np.random.randn(n1, n2)
        else:
            # Extend orthonormal matrix E to orthonormal matrix with given shape
            n1 -= E.shape[0]
            E_on = np.r_[E, np.random.randn(n1, n2)]
        self.gram_schmidt(E_on, VS, n1)
        return E_on

    def householder_triangularization(self, A_, VS, R=None, E=None, V=None, returns=False):

        (Sequentially) compute the upper triangular matrix of a QR-decomposition
        of a matrix A_. 

        Parameters
        ----------
        A_ : np.ndarray
            Snapshot matrix.
        VS : VectorSpace
            Vector space object.
        R : None or np.ndarray
            Upper triangular matrix (N_R x N_R) obtained from the Householder
            triangularization of the first N_R columns in A_.
        E : np.ndarray
            Orthonormal matrix created in Householder triangularization.
        V : np.ndarray
            Householder matrix created in Householder triangularization.
        returns : bool
            If True, return E and V in addition to R.

        Returns
        -------
        R : np.ndarray
            Upper triangular matrix R of the QR-decomposition of A_.
        (E) : np.ndarray
            Orthonormal matrix created in Householder triangularization.
        (V) : np.ndarray
            Householder matrix created in Householder triangularization.

        References
        ----------
        [1] Lloyd N. Trefethen: Householder triangularization of a quasimatrix.
            IMA Journal of Numerical Analysis (2008). DOI: 10.1093/imanum/dri017

        A = A_.copy()
        N_A = A.shape[0]

        # Declare matrix R or extend it with zeros if it already exists
        if R is None:
            N_R = 0
            R = np.zeros((N_A, N_A)) 
        else:
            N_R = R.shape[0]
            R = np.pad(R, (0, N_A-N_R), mode='constant')

        # Get (or extend) an orthonormal matrix of the same shape as snapshot matrix
        E = self.get_orthonormal_matrix(A.shape, VS, E)

        # Declare matrix V for Householder vectors or extend it if it already exists
        if V is None:
            V = np.empty((N_A, A.shape[1]))
        else:
            V = np.pad(V, ((0, N_A-N_R), (0, 0)), mode='constant')

        for j in range(N_R, N_A):
            # Apply the reflection to j-th snapshot
            for k in range(j):
                A[j] -= 2 * V[k] * VS.inner_product(V[k], A[j])
                R[k, j] = VS.inner_product(E[k], A[j])
                A[j] -= E[k] * R[k, j]

            R[j, j] = VS.norm(A[j])

            # Modify E to take account of sign
            alpha = VS.inner_product(E[j], A[j])
            if abs(alpha) > 1e-17:
                E[j] *= - alpha / abs(alpha)

            # Vector defining next reflection
            V[j] = R[j, j] * E[j] - A[j]
            for i in range(j):
                V[j] -= VS.inner_product(E[i], V[j]) * E[i]

            # If zero vector, reflection is j-th column in orthonormal matrix
            sigma = VS.norm(V[j])
            if abs(sigma) > 1e-17:
                V[j] /= sigma
            else:
                V[j] = E[j]

        if returns:
            return R, E, V
        return R
"""
    