"""
PECwg.py
-------

A waveguide is a medium separated from its environment by
a set of perfectly conducting walls, whereof one acts as
an inlet for an incoming electromagnetic wave.
"""

import fenics as fen

def create(mesh, inlet):
    # Create a boundary function evaluating to 1 for inlet and 2 for wall
    boundary = fen.MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
    inlet.mark(boundary, 1)
    class Wall(fen.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and not inlet.inside(x, 'on_boundary')
    Wall().mark(boundary, 2)
    if mesh.topology().dim() == 2:
        V = fen.FunctionSpace(mesh, 'P', 1)
        bc = fen.DirichletBC(V, fen.Constant(0.0), boundary, 2)
    else:
        V = fen.FunctionSpace(mesh, 'N1curl', 1)
        bc = fen.DirichletBC(V, fen.Constant((0.0, 0.0, 0.0)), boundary, 2)
    return boundary, V, bc

def setup(boundary, g_inlet, mu, eps, V, bc=None):    
    A = fen.TrialFunction(V)
    v = fen.TestFunction(V)

    # Neumann boundary condition for linear form
    ds = fen.Measure('ds', subdomain_data=boundary)
    L = fen.assemble(fen.dot(g_inlet, v)*ds(1))

    # Neumann boundary condition for bilinear form
    if V.mesh().topology().dim() == 2:
        K = fen.assemble((1 / mu)*fen.dot(fen.grad(A), fen.grad(v))*fen.dx)
    else:
        K = fen.assemble((1 / mu)*fen.dot(fen.curl(A), fen.curl(v))*fen.dx)
    M = fen.assemble(eps*fen.dot(A, v)*fen.dx)
    
    if bc is not None:
        bc.apply(K)
        bc.zero(M)
        #bc.apply(L) <- usually not necessary...

    return K, M, L

def solve(omega, K, M, L, V):  
    a = K - omega**2 * M
    A = fen.Function(V)
    fen.solve(a, A.vector(), L)
    return A