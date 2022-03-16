"""
helpers.py
----------

Collection of helper functions. 
"""

import fenics as fen
import PECwg

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

import matplotlib.pyplot as plt


"""Exportations"""

def export_form_as_sparse_matrix(a, fname):
    """
    Export form 'a' as sparse matrix in the format [i, j, a(i, j)].
    """
    M = fen.assemble(a)
    Mmat = fen.as_backend_type(M).mat()
    mat = scipy.sparse.coo_matrix(scipy.sparse.csr_matrix(Mmat.getValuesCSR()[::-1]))

    np.savetxt(fname + '.txt', np.c_[mat.row, mat.col, mat.data],
               fmt=['%d', '%d', '%.16f']) 

def export_field_at_vertex_coordinates(A, mesh, fname):
    """
    Export (vector) field 'A' evaluated at vertex coordinates '(x1, x2, ...)' in
    the format [A_1(x1, x2, ...), A_2(x1, x2, ...), ..., x1, x2, ...].
    """
    coords = mesh.coordinates()
    n_coords = coords.shape[0]

    Avec = A.compute_vertex_values(mesh)
    Acoords = Avec.reshape((n_coords, int(len(Avec)/n_coords)), order='F')
    np.savetxt(fname + '.txt', np.c_[Acoords, coords], fmt = '%.16f')

def export_field_as_png_plot(A, fname):
    plt.figure()
    plt.colorbar(fen.plot(A))
    plt.savefig(fname + '.png', dpi=100)
    
def export_field_as_pvd_plot(A, fname):
    file = fen.File(fname + '.pvd')
    file << A


"""Waveguides"""
    
def get_2d_analytical_resonant_frequencies(Lx, Ly, a, b):
    freqs = lambda n, m: np.pi*pow(n**2/(Lx)**2 + (m+0.5)**2/(Ly)**2 , 0.5)
    n_max = np.ceil(b * Lx / np.pi).astype('int')
    m_max = np.ceil(b * Ly / np.pi - 0.5).astype('int')
    eigs = np.unique(np.frompyfunc(freqs, 2, 1).outer(range(1, n_max+1), range(m_max+1)))
    return [e for e in eigs if a <= e and e <= b]

def get_3d_analytical_eigenvalues(Lx, Ly, Lz, a, b):
    freqs = lambda n, m, l: np.pi*pow(n**2/(Lx)**2 + m**2/(Ly)**2 + (l+0.5)**2/(Lz)**2 , 0.5)
    n_max = np.ceil(b * Lx / np.pi).astype('int')
    m_max = np.ceil(b * Ly / np.pi).astype('int')
    l_max = np.ceil(b * Lz / np.pi - 0.5).astype('int')
    eigs_nml = np.empty((n_max, m_max+1, l_max+1))
    for n in range(1, n_max+1):
        for m in range(m_max+1):
            for l in range(l_max+1):
                eigs_nml[n-1, m, l] = freqs(n, m, l)
    eigs = np.unique(eigs_nml)
    return [x for x in eigs if a <= x and x <= b]

def solve_eigenvalue_problem(K, M, V, bc, a, b, k=10):
    boundary_points = bc.get_boundary_values().keys()
    all_points = V.dofmap().dofs()
    inner_points = list(set(all_points) - set(boundary_points))
    # Alternative 1: inner_points = np.where(~np.all(np.isclose(M.array(), 0), axis=1))[0]
    # Alternative 2: ones = fen.assemble(fen.TrialFunction(V)*fen.dx)
    #                ones[:] = 1
    #                bc.apply(ones)
    #                inner_points = np.array(ones.get_local(), dtype=bool)

    Mmat = fen.as_backend_type(M).mat().getValuesCSR()[::-1]
    M_sparse = scipy.sparse.csr_matrix(Mmat)
    M_reduced = M_sparse[inner_points, :][:, inner_points]
    
    Kmat = fen.as_backend_type(K).mat().getValuesCSR()[::-1]
    K_sparse = scipy.sparse.csr_matrix(Kmat)
    K_reduced = K_sparse[inner_points, :][:, inner_points]
    
    eigvals, eigvecs = scipy.sparse.linalg.eigsh(K_reduced, k=k, M=M_reduced, sigma=(a+b)/2)

    eigvals_inside = [e1 for e1 in eigvals if a <= e1 and e1 <= b]
    eigvecs_inside = [e2 for e1, e2 in zip(eigvals, eigvecs.T) if a <= e1 and e1 <= b]

    if len(eigvals_inside) == k:
        print(f'WARNING: Found exactly {k} eigenvalues within [{a}, {b}].')
        print('Increase parameter "k" to make sure all eigenvalues are found.')
    
    return eigvals_inside, eigvecs_inside

def insert_boundary_values(field, V, bc):
    boundary_indices = list(bc.get_boundary_values().keys())
    boundary_values = list(bc.get_boundary_values().values())
    num_indices = len(V.dofmap().dofs())
    inner_indices = list(set(range(num_indices)) - set(boundary_indices))

    field_inserted = np.empty(num_indices)
    field_inserted[inner_indices] = field
    field_inserted[boundary_indices] = boundary_values
    
    return field_inserted

def plot_g_z_inlet(g_z_inlet, inlet, V):
    points = V.tabulate_dof_coordinates()
    inlet_points = np.array([p for p in points if inlet.inside(p, 'on_boundary')])
    points = inlet_points[inlet_points[:, 1].argsort()]
    g_z_interp = fen.interpolate(g_z_inlet, V)
    g_z_points = np.array([g_z_interp(point) for point in points])
    plt.plot(points[:, 1], g_z_points)
    plt.xlabel('Gridpoints (y-component)')
    plt.ylabel('Input field g_z_inlet')
    plt.margins(x=0, y=0)

def plot_2d_field(field, V, bc, reduced=True):
    if reduced:
        field = insert_boundary_values(field, V, bc)
    
    coords = V.tabulate_dof_coordinates().reshape((-1, 2))
    fig, ax = plt.subplots(figsize=(6, 6*np.ptp(coords[:, 1])/np.ptp(coords[:, 0])))
    ax.margins(x=0, y=0)
    ax.tripcolor(coords[:, 0], coords[:, 1], field, shading='gouraud')

def eigvec_dot_L(eigvec, V, bc, L):
    full_eigvec = insert_boundary_values(eigvec, V, bc)
    return np.inner(full_eigvec, L.get_local())

def get_solution_L2_norms(omegas, K, M, L, V):
    L2_norms = np.empty(len(omegas))
    F = fen.assemble(fen.dot(fen.TrialFunction(V), fen.TestFunction(V))*fen.dx)
    for i, omega in enumerate(omegas):
        A_z = PECwg.solve(omega, K, M, L, V).vector()
        L2_norms[i] = pow((A_z*(F*A_z)).sum(), 0.5)
    return L2_norms

def plot_solution_L2_norms(omegas, K, M, L, V, bc):
    L2_norms =  get_solution_L2_norms(omegas, K, M, L, V)

    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(omegas, L2_norms)
    ax[0].set_yscale('log')
    ax[0].set_ylabel('L2-norm of solution')

    eigvals, eigvecs = solve_eigenvalue_problem(K, M, V, bc, omegas[0]**2, omegas[-1]**2, k=20)

    ax[0].vlines(np.sqrt(eigvals), 0, np.max(L2_norms), linewidth=0.5, colors='k', alpha=0.5)
    ax[0].set_xticks(np.sqrt(eigvals), minor=True)
    ax[0].margins(x=0, y=0)

    dotproducts = [eigvec_dot_L(eigvec, V, bc, L) for eigvec in eigvecs]    

    ax[1].bar(np.sqrt(eigvals), np.abs(dotproducts), width=0.05)
    ax[1].set_yscale('log')
    ax[1].set_xlabel('Frequency omega')
    ax[1].set_ylabel('Dotproduct |<u, L>|')
    ax[1].set_xlim(omegas[0], omegas[-1])
    ax[1].margins(y=0)
    return eigvals, eigvecs, dotproducts