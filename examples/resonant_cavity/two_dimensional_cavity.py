# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

import fenics as fen

from context import src
from src.time_harmonic_maxwell_problem import TimeHarmonicMaxwellProblem

class TwoDimensionalCavity(TimeHarmonicMaxwellProblem):
    def __init__(self, mesh, B_N, g_N):
        V = fen.FunctionSpace(mesh, 'P', 1)

        mu = fen.Expression('1.0', degree=2)
        eps = fen.Expression('1.0', degree=2)
        j = fen.Expression('0.0', degree=2)

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

    def plot_solution_trace(self, trace, axis=1):
        A_sol = self.get_solution(tonumpy=False)
        for i, A in enumerate(A_sol):
            plt.figure()
            plt.title('Solution on trace at frequency \u03C9 = {:.3f} rad/s'.format(self.omega[i]))
            all_coords = self.V.tabulate_dof_coordinates()
            trace_coords = np.array([x for x in all_coords if trace.inside(x, 'on_boundary')])
            A_trace = [A(x) for x in trace_coords]
            xs, ys = zip(*sorted(zip(trace_coords[:, axis], A_trace)))
            plt.plot(xs, ys)
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
