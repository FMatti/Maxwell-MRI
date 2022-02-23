"""
helpers.py
----------

Collection of helper functions. 
"""

from fenics import *
from mshr import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix, csr_matrix

def export_form_as_sparse_matrix(a, fname):
    """
    Export form 'a' as sparse matrix in the format [i, j, a(i, j)].
    """
    M = assemble(a)
    Mmat = as_backend_type(M).mat()
    mat = coo_matrix(csr_matrix(Mmat.getValuesCSR()[::-1]))

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
    plt.colorbar(plot(A))
    plt.savefig(fname + '.png', dpi=100)
