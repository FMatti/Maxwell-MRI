"""
waveguide_2D.py
----------

I think of a waveguide as a medium separated from its environment
by a set of walls, whereof one acts as an inlet for an incoming
electromagnetic wave.
"""

import fenics as fen
import mshr
import helpers
import numpy as np

height = 1.0 # radius of wire
width = 5.0 # radius of domain
mu = 1.0 # 4*np.pi*1e-7 # magnetic permeability inside waveguide
eps = 1.0 # 8.854187e-12 # electric permittivity inside waveguide

B_inlet = fen.Expression('1.0', degree=2) # Constant input wave
#B_inlet = fen.Expression('exp(-pow(x[1] - center, 2) / 0.01)', degree=2, center=height/2) # Gaussian input wave
f = width / (np.pi*2) # frequency of input wave

domain = mshr.Rectangle(fen.Point(0.0, 0.0), fen.Point(width, height)) \
       + mshr.Rectangle(fen.Point(width/4, height), fen.Point(width/2, height*5/4)) # Additional "cubby" in waveguide
mesh = mshr.generate_mesh(domain, 128)

V = fen.FunctionSpace(mesh, 'P', 1)
A_z = fen.TrialFunction(V)
v_z = fen.TestFunction(V)

dx = fen.Measure('dx', domain=mesh)
a_stiff = (1 / mu)*fen.dot(fen.grad(A_z), fen.grad(v_z))*dx
a_mass = eps*A_z*v_z*dx
a = a_stiff - (2*np.pi*f)**2 * a_mass

class Inlet(fen.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and fen.near(x[0], 0)
    
class UpWall(fen.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and fen.near(x[1], height)
    
class DownWall(fen.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and fen.near(x[1], 0)
    
class Outlet(fen.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and fen.near(x[0], width)
    
class Cubby(fen.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and fen.between(x[1], [height, height*5/4]) and fen.between(x[0], [width/4, width/2])

# Marking boundaries
boundary = fen.MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
inlet = Inlet()
inlet.mark(boundary, 1)
upwall = UpWall()
upwall.mark(boundary, 2)
downwall = DownWall()
downwall.mark(boundary, 3)
outlet = Outlet()
outlet.mark(boundary, 4)
cubby = Cubby()
cubby.mark(boundary, 5)
ds = fen.Measure('ds', subdomain_data=boundary)

# Neumann boundary condition
g_z = fen.Expression('-(1/mu)*B_inlet', degree=2, B_inlet=B_inlet, mu=mu)
L = g_z*v_z*ds(1)

# Dirichlet boundary conditions
bc_upwall = fen.DirichletBC(V, fen.Constant('0.0'), upwall)
bc_downwall = fen.DirichletBC(V, fen.Constant('0.0'), downwall)
bc_outlet = fen.DirichletBC(V, fen.Constant('0.0'), outlet)
bc_cubby = fen.DirichletBC(V, fen.Constant('0.0'), cubby)
bcs = [bc_upwall, bc_downwall, bc_outlet, bc_cubby]

A_z = fen.Function(V)
fen.solve(a == L, A_z, bcs)

helpers.export_form_as_sparse_matrix(a_stiff, 'a_stiff')
helpers.export_form_as_sparse_matrix(a_mass, 'a_mass')
helpers.export_field_at_vertex_coordinates(A_z, mesh, 'A_vertex')
helpers.export_field_as_png_plot(A_z, 'waveguide_2D')