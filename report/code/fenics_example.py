
import numpy as np
import fenics as fen

# Get a unit cube mesh
nx, ny, nz = 10, 10, 10
mesh = fen.UnitCubeMesh(nx, ny, nz)

# Function space using Nedelec elements of the first kind
V = fen.FunctionSpace(mesh, 'N1curl', 1)

# Define inlet subdomain
class Inlet(fen.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and fen.near(x[0], 0)

# Define perfectly electrically conducting wall subdomain
class PECWalls(fen.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and not Inlet().inside(x, on_boundary)

# Identify each boundary type with an id
boundary_id = fen.MeshFunction('size_t', mesh, mesh.topology().dim()-1)
boundary_id.set_all(0)
Inlet().mark(boundary_id, 1)
PECWalls().mark(boundary_id, 2)

# Dirichlet boundary condition
u_D = fen.Expression(('0.0', '0.0', '0.0'), degree=2)
bc = fen.DirichletBC(V, u_D, boundary_id, 2)

# Neumann boundary integral term and boundary measure
g_N = fen.Expression(('0.0', '0.0', '1.0'), degree=2)
ds = fen.Measure('ds', subdomain_data=boundary_id)

# Trial and test functions
u = fen.TrialFunction(V)
v = fen.TestFunction(V)

# Neumann boundary integral term
N = fen.assemble(fen.dot(g_N, v) * ds(2))

# Stiffness matrix
K = fen.assemble(fen.dot(fen.curl(u), fen.curl(v)) * fen.dx)
bc.apply(K)

# Mass matrix
M = fen.assemble(fen.dot(u, v) * fen.dx)
bc.zero(M)

# L2-norm function
def L2_norm(u):
    u_vec = u.vector().get_local()
    return pow(((M * u_vec) * u_vec).sum(), 0.5)

# Solution at a certain frequency
omegas = np.linspace(6.2, 6.8, 200)
norms = []
u = fen.Function(V)
for omega in omegas:
    fen.solve(K - omega**2 * M, u.vector(), N)
    norms.append(L2_norm(u))

# Plot the L2-norms
plt.plot(omegas, norms)
plt.yscale('log')
