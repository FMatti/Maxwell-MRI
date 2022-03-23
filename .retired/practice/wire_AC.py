"""
wire_AC.py
----------

Time-harmonic frequency response of an infinitely extending circular wire
carrying a sinusoidal alternating current.
"""

from fenics import *
from mshr import *
from helpers import *

f = 1000 # frequency of alternating current
current = 10000 # amplitude of current flowing through wire

r = 0.1 # radius of wire
R = 1.0 # radius of domain
mu_wire = 1.256629e-6 # magnetic permeability of the wire
mu_vacuum = 4*np.pi*1e-7 # magnetic permeability outside the wire

domain = Circle(Point(0, 0), R)
mesh = generate_mesh(domain, 32)

V = FunctionSpace(mesh, 'P', 1)
A_z = TrialFunction(V)
v_z = TestFunction(V)
bc = DirichletBC(V, Constant(0.0), 'on_boundary')

dx = Measure('dx', domain=mesh)
mu = Expression('x[0]*x[0] + x[1]*x[1] < r*r ? mu_wire : mu_vacuum',
                degree=2, r=r, mu_wire=mu_wire, mu_vacuum=mu_vacuum) 
a_stiff = (1 / mu)*dot(grad(A_z), grad(v_z))*dx
a_mass = A_z*v_z*dx
a = a_stiff - (2*np.pi*f)**2 * a_mass
J = Expression('x[0]*x[0] + x[1]*x[1] < r*r ? current : 0',
               degree=2, r=r, current=current)
L = J*v_z*dx

A_z = Function(V)
solve(a == L, A_z, bc)

export_form_as_sparse_matrix(a_stiff, 'a_stiff')
export_form_as_sparse_matrix(a_mass, 'a_mass')
export_field_at_vertex_coordinates(A_z, mesh, 'A_vertex')
export_field_as_png_plot(A_z, 'wire_AC')
