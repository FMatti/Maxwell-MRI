"""
wg2D.py
-------

A waveguide is a medium separated from its environment by
a set of perfectly conducting walls, whereof one acts as
an inlet for an incoming electromagnetic wave.
"""

import fenics as fen
import mshr
import helpers
import numpy as np

def create(mesh, inlet):
    # Create a boundary function evaluating to 1 for inlet and 2 for wall
    boundary = fen.MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
    inlet.mark(boundary, 1)
    class Wall(fen.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and not inlet.inside(x, 'on_boundary')
    Wall().mark(boundary, 2)
    
    V = fen.FunctionSpace(mesh, 'P', 1)
    
    # Dirichlet boundary condition
    bc = fen.DirichletBC(V, fen.Constant('0.0'), boundary, 2)
    
    return boundary, V, bc

def setup(boundary, g_z_inlet, mu, eps, V, bc=None):    
    A_z = fen.TrialFunction(V)
    v_z = fen.TestFunction(V)

    # Neumann boundary condition for linear form
    ds = fen.Measure('ds', subdomain_data=boundary)
    L = fen.assemble(g_z_inlet*v_z*ds(1))

    # Neumann boundary condition for bilinear form
    K = fen.assemble((1 / mu)*fen.dot(fen.grad(A_z), fen.grad(v_z))*fen.dx)
    M = fen.assemble(eps*A_z*v_z*fen.dx)
    
    if bc is not None:
        bc.apply(K)
        bc.zero(M)
        bc.apply(L)
    
    return K, M, L

def solve(omega, K, M, L, V):  
    a = K - omega**2 * M  
    A_z = fen.Function(V)
    fen.solve(a, A_z.vector(), L)
    return A_z
