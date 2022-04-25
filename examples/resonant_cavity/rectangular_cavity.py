# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

import fenics as fen

from context import src
from src.time_harmonic_maxwell_problem import TimeHarmonicMaxwellProblem

class RectangularCavity(TimeHarmonicMaxwellProblem):
    def __init__(self, Lx, Ly, Nx, Ny, g_N):
        self.Lx = Lx
        self.Ly = Ly
        self.Nx = Nx
        self.Ny = Ny
        mesh = fen.RectangleMesh(fen.Point(0.0, 0.0), fen.Point(self.Lx, self.Ly), Nx, Ny, 'crossed')
        V = fen.FunctionSpace(mesh, 'P', 1)

        mu = fen.Expression('1.0', degree=2)
        eps = fen.Expression('1.0', degree=2)
        j = fen.Expression('0.0', degree=2)

        class B_N(fen.SubDomain):
            def inside(self_, x, on_boundary):
                return on_boundary and fen.near(x[0], 0.0) and x[1]>0.0 and x[1]<self.Ly

        class B_D(fen.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and not B_N().inside(x, 'on_boundary')

        A_D = fen.Expression('0.0', degree=2)

        TimeHarmonicMaxwellProblem.__init__(self, V, mu, eps, j, B_D(), B_N(), A_D, g_N)

    def plot_solution(self):
        A_sol = self.get_solution(tonumpy=False)
        for i, A in enumerate(A_sol):
            plt.figure()
            plt.title('Solution to system at frequency \u03C9 = {:.3f} rad/s'.format(self.omega[i]))
            fig = fen.plot(A)
            plt.colorbar(fig, orientation='horizontal')
            plt.show()

    def plot_solution_trace(self, trace):
        A_sol = self.get_solution(tonumpy=False)
        for i, A in enumerate(A_sol):
            plt.figure()
            plt.title('Solution on trace at frequency \u03C9 = {:.3f} rad/s'.format(self.omega[i]))
            all_coords = self.V.tabulate_dof_coordinates()
            trace_coords = np.array([x for x in all_coords if trace.inside(x, 'on_boundary')])
            A_trace = [A(x) for x in trace_coords]
            plt.plot(trace_coords[:, 1], A_trace)
            plt.show()

    def plot_external_solution(self, A_vec, contains_boundary_values=False, omega=None):
        plt.figure()
        if omega is not None:
            plt.title('Solution to system at frequency \u03C9 = {:.3f} rad/s'.format(omega))
        A_func = fen.Function(self.V)
        if not contains_boundary_values:
            A_vec = self.insert_boundary_values(A_vec)
        A_func.vector()[:] = A_vec
        fig = fen.plot(A_func)
        plt.colorbar(fig, orientation='horizontal')
        plt.show()

    def plot_g_N(self):
        plt.figure()
        all_coords = self.V.tabulate_dof_coordinates()
        inlet_coords = np.array([x for x in all_coords if self.B_N.inside(x, 'on_boundary')])
        g_N_coords = [self.g_N(x) for x in inlet_coords]
        plt.plot(inlet_coords[:, 1], g_N_coords)
        plt.ylabel('g_N')
        plt.xlim(0.0, self.Ly)
        plt.show()

    def get_analytical_eigenfrequencies(self, a, b):
        freqs = lambda n, m: np.pi*pow(((n+0.5)/self.Lx)**2 + (m/self.Ly)**2, 0.5)
        n_max = np.ceil(b * self.Lx / np.pi - 0.5).astype('int')
        m_max = np.ceil(b * self.Ly / np.pi).astype('int')
        eigs = np.unique(np.frompyfunc(freqs, 2, 1).outer(range(n_max+1), range(1, m_max+1)))
        return [e for e in eigs if a <= e and e <= b]
