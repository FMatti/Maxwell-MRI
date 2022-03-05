"""
wg2D.py
-----------------------

I think of a waveguide as a medium separated from its environment
by a set of walls, whereof one acts as an inlet for an incoming
electromagnetic wave.
"""

import fenics as fen
import mshr
import helpers
import numpy as np

def create(domain, inlet, resolution=64):
    mesh = mshr.generate_mesh(domain, resolution)

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
    
    return mesh, boundary, V, bc

def setup(mesh, boundary, g_z_inlet, mu, eps, V):    
    A_z = fen.TrialFunction(V)
    v_z = fen.TestFunction(V)

    # Neumann boundary condition for linear form
    ds = fen.Measure('ds', subdomain_data=boundary)
    L = g_z_inlet*v_z*ds(1)

    # Neumann boundary condition for bilinear form
    dx = fen.Measure('dx', domain=mesh)
    K = (1 / mu)*fen.dot(fen.grad(A_z), fen.grad(v_z))*dx
    M = eps*A_z*v_z*dx
    
    return K, M, L

def solve(omega, K, M, L, bc, V):  
    a = K - omega**2 * M  
    A_z = fen.Function(V)
    fen.solve(a == L, A_z, bc)
    return A_z
