import os
import sys
import numpy as np
import fenics as fen
import mshr
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath('__file__')))))
from src.time_harmonic_maxwell_problem import TimeHarmonicMaxwellProblem

class CuboidWaveguide(TimeHarmonicMaxwellProblem):
    def __init__(self, Lx, Ly, Lz, Nx, Ny, Nz, g_N):
        self.Lx, self.Ly, self.Lz = Lx, Ly, Lz
        mesh = fen.BoxMesh(fen.Point(0.0, 0.0, 0.0), fen.Point(self.Lx, self.Ly, self.Lz), Nx, Ny, Nz)
        V = fen.FunctionSpace(mesh, 'N1curl', 1)

        mu = fen.Expression('1.0', degree=2)
        eps = fen.Expression('1.0', degree=2)
        j = fen.Expression(('0.0', '0.0', '0.0'), degree=2)

        class B_N(fen.SubDomain):
            def inside(self_, x, on_boundary):
                return on_boundary and fen.near(x[0], 0.0) and x[1]>0.0 and x[1]<self.Ly and x[2]>0.0 and x[2]<self.Lz

        class B_D(fen.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and not B_N().inside(x, 'on_boundary')

        A_D = fen.Expression(('0.0', '0.0', '0.0'), degree=2)

        TimeHarmonicMaxwellProblem.__init__(self, V, mu, eps, j, B_D(), B_N(), A_D, g_N)

    def plot_solution(self):
        plt.figure()
        plt.title(f'Solution to system at frequency \u03C9 = {self.omega} rad/s')
        fig = fen.plot(self.A_sol)
        plt.show()

    def plot_external_solution(self, A_vec, contains_boundary_values=False, omega=None):
        plt.figure()
        if omega is not None:
            plt.title(f'Solution to system at frequency \u03C9 = {omega} rad/s')
        A_func = fen.Function(self.V)
        if not contains_boundary_values:
            A_vec = self.insert_boundary_values(A_vec)
        A_func.vector()[:] = A_vec
        fen.plot(A_func)
        fig = fen.plot(self.A_sol)
        plt.show()

    def plot_g_N(self):
        plt.figure()
        all_coords = self.V.tabulate_dof_coordinates()
        inlet_coords = np.array([x for x in all_coords if self.B_N.inside(x, 'on_boundary')])
        g_N_coords = np.array([self.g_N(x) for x in inlet_coords])
        fig, ax = plt.subplots()
        ax.margins(x=0, y=0)
        ax.tripcolor(inlet_coords[:, 1], inlet_coords[:, 2], g_N_coords[:, 0], shading='gouraud')
        plt.xlabel('y')
        plt.ylabel('z')
        plt.show()
 
    def get_analytical_eigenfrequencies(self, a, b):
        freqs = lambda n, m, l: np.pi*pow(((n+0.5)/self.Lx)**2 + (m/self.Ly)**2 + (l/self.Lz)**2, 0.5)
        n_max = np.ceil(b * self.Lx / np.pi - 0.5).astype('int')
        m_max = np.ceil(b * self.Ly / np.pi).astype('int')
        l_max = np.ceil(b * self.Lz / np.pi).astype('int')
        eigs_nml = np.empty((n_max+1, m_max, l_max+1))
        for n in range(n_max+1):
            for m in range(1, m_max+1):
                for l in range(l_max+1):
                    eigs_nml[n, m-1, l] = freqs(n, m, l)
        eigs = np.unique(eigs_nml)
        return [x for x in eigs if a <= x and x <= b]