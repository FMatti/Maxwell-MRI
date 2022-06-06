# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

import fenics as fen

from context import src
from src.time_harmonic_maxwell_problem import TimeHarmonicMaxwellProblem

class TwoDimensionalCavity(TimeHarmonicMaxwellProblem):
    """
    Daughter class of TimeHarmonicMaxwellProblem specialized on general
    two-dimensional cavities.

    Members
    -------
    (Additional members inherited from TimeHarmonicMaxwellProblem)

    Methods
    -------
    plot_solution() : None -> None
        Plot the solution in a 2d colormap.
    plot_solution_trace(trace) : fen.Subdomain -> None
        Plot the solution along a trace.
    plot_external_solution(u_vec) : np.ndarray -> None
        Plot an external solution in a 2d colormap.
    (Additional methods inherited from TimeHarmonicMaxwellProblem)

    Usage
    -----
    Rectangular 5x1 cavity with mesh of resolution 100, Neumann boundary (inlet)
    at x=0 and Neumann datum 1. 
    >>> import mshr
    >>> Lx, Ly = 5.0, 1.0
    >>> domain = mshr.Rectangle(fen.Point(0.0, 0.0), fen.Point(Lx, Ly))
    >>> mesh = mshr.generate_mesh(domain, 100)
    >>>
    >>> class inlet(fen.SubDomain):
    >>>     def inside(self, x, on_boundary):
    >>>         return on_boundary and fen.near(x[0], 0.0) and x[1]>0.0 and x[1]<Ly
    >>>
    >>> g_N = fen.Expression('1.0', degree=2)
    >>> TDC = TwoDimensionalCavity(mesh=mesh, B_N=inlet, g_N=g_N)
    >>> TDC.setup()
    """
    def __init__(self, mesh, B_N, g_N):
        V = fen.FunctionSpace(mesh, 'P', 1)

        # Trivial physical constants
        mu = fen.Expression('1.0', degree=2)
        eps = fen.Expression('1.0', degree=2)
        j = fen.Expression('0.0', degree=2)

        # Dirichlet boundary for all non-Neumann boundaries
        class B_D(fen.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and not B_N().inside(x, 'on_boundary')

        u_D = fen.Expression('0.0', degree=2)

        TimeHarmonicMaxwellProblem.__init__(self, V, mu, eps, j, B_D(), u_D, B_N(), g_N)

    def plot_solution(self):
        """Plot the solution in a 2d colormap"""
        solution = self.get_solution()
        for i, u in enumerate(solution):
            plt.figure()
            plt.title('Solution to system at frequency \u03C9 = {:.3f} rad/s'.format(self.omega[i]))
            fig = fen.plot(u)
            plt.colorbar(fig, orientation='horizontal')
            plt.show()

    def plot_solution_trace(self, trace, axis=1):
        """Plot solution along a trace"""
        solution = self.get_solution()
        for i, u in enumerate(solution):
            plt.figure()
            plt.title('Solution on trace at frequency \u03C9 = {:.3f} rad/s'.format(self.omega[i]))
            all_coords = self.V.tabulate_dof_coordinates()
            trace_coords = np.array([x for x in all_coords if trace.inside(x, 'on_boundary')])
            u_trace = [u(x) for x in trace_coords]
            xs, ys = zip(*sorted(zip(trace_coords[:, axis], u_trace)))
            plt.plot(xs, ys)
            plt.show()

    def plot_external_solution(self, u_vec, contains_boundary_values=False, omega=None):
        """Plot the external solution u_vec in a 2d colormap"""
        plt.figure()
        if omega is not None:
                        plt.title('Solution to system at frequency \u03C9 = {:.3f} rad/s'.format(omega))

        # Insert Dirichlet boundary values if they were are not already contained
        u_func = fen.Function(self.V)
        if not contains_boundary_values:
            u_vec = self.insert_boundary_values(u_vec)
        u_func.vector()[:] = u_vec
        fig = fen.plot(u_func)
        plt.colorbar(fig, orientation='horizontal')
        plt.show()
