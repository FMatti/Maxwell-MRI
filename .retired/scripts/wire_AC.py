"""
wire_AC.py
----------

Time-harmonic frequency response of an infinitely extending circular wire
carrying a sinusoidal alternating current.
"""

import fenics as fen
import mshr
import helpers
import numpy as np

f = 1000 # frequency of alternating current
current = 10000 # amplitude of current flowing through wire

r = 0.1 # radius of wire
R = 1.0 # radius of domain
mu_wire = 1.256629e-6 # magnetic permeability of the wire
mu_vacuum = 4*np.pi*1e-7 # magnetic permeability outside the wire
eps_wire = 2 # electric permittivity of the wire
eps_vacuum = 1 # electric permittivity outside the wire

domain = mshr.Circle(fen.Point(0, 0), R)
mesh = mshr.generate_mesh(domain, 100)

V = fen.FunctionSpace(mesh, 'P', 1)
A_z = fen.TrialFunction(V)
v_z = fen.TestFunction(V)
bc = fen.DirichletBC(V, fen.Constant(0.0), 'on_boundary')

dx = fen.Measure('dx', domain=mesh)
eps = fen.Expression('x[0]*x[0] + x[1]*x[1] < r*r ? eps_wire : eps_vacuum',
                degree=2, r=r, eps_wire=eps_wire, eps_vacuum=eps_vacuum) 
mu = fen.Expression('x[0]*x[0] + x[1]*x[1] < r*r ? mu_wire : mu_vacuum',
                degree=2, r=r, mu_wire=mu_wire, mu_vacuum=mu_vacuum) 
a_stiff = (1 / mu)*fen.dot(fen.grad(A_z), fen.grad(v_z))*dx
a_mass = eps*A_z*v_z*dx
a = a_stiff - (2*np.pi*f)**2 * a_mass
J = fen.Expression('x[0]*x[0] + x[1]*x[1] < r*r ? current : 0',
               degree=2, r=r, current=current)
L = J*v_z*dx

A_z = fen.Function(V)
fen.solve(a == L, A_z, bc)

helpers.export_form_as_sparse_matrix(a_stiff, 'a_stiff')
helpers.export_form_as_sparse_matrix(a_mass, 'a_mass')
helpers.export_field_at_vertex_coordinates(A_z, mesh, 'A_vertex')
helpers.export_field_as_png_plot(A_z, 'wire_AC')
