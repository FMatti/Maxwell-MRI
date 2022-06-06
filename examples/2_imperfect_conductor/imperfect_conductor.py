# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

import fenics as fen

from context import src
from src.time_harmonic_maxwell_problem import TimeHarmonicMaxwellProblem

class ImperfectConductor(TimeHarmonicMaxwellProblem):
    """
    Daughter class of TimeHarmonicMaxwellProblem specialized on rectangular
    cavities with an imperfectly conducting (impedance) boundary.

                                 L_x
            <------------------------------------------->
        I   ____________________________________________  I  ^
        n   >                                          X  m  |
        l   >                                          X  p  | L_y
        e   >                                          X  e  |
        t   ____________________________________________  d  v

    Members
    -------
    Lx : float
        Length of the cavity in x-direction.
    Ly : float
        Length of the cavity in y-direction.
    Nx : int
        Number of subdivisions of domain in x-direction.
    Ny : int
        Number of subdivisions of domain in y-direction.
    (Additional members inherited from TimeHarmonicMaxwellProblem)

    Methods
    -------
    plot_solution() : None -> None
        Plot the solution in a 2d colormap.
    plot_solution_trace(trace) : fen.Subdomain -> None
        Plot the solution along a trace.
    plot_external_solution(u_vec) : np.ndarray -> None
        Plot an external solution in a 2d colormap.
    plot_g_N() : None -> None
        Plot the Neumann boundary condition.
    (Additional methods inherited from TimeHarmonicMaxwellProblem)

    Usage
    -----
    Rectangular 5x1 cavity with 101x21 subdivisions, Neumann (inlet) datum 1,
    and impedance 1 at the imperfectly conducting boundary.
    >>> Lx, Ly = 5.0, 1.0
    >>> Nx, Ny = 101, 21
    >>> g_N = fen.Expression('1.0', degree=2)
    >>> imp = fen.Expression('1.0', degree=2)
    >>> RC = ImperfectConductor(Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny, g_N=g_N, imp=imp)
    >>> RC.setup()
    """
    def __init__(self, Lx, Ly, Nx, Ny, g_N, imp):
        self.Lx = Lx
        self.Ly = Ly
        self.Nx = Nx
        self.Ny = Ny
        mesh = fen.RectangleMesh(fen.Point(0.0, 0.0), fen.Point(self.Lx, self.Ly), Nx, Ny, 'crossed')
        V = fen.FunctionSpace(mesh, 'P', 1)

        # Trivial physical constants
        mu = fen.Expression('1.0', degree=2)
        eps = fen.Expression('1.0', degree=2)
        j = fen.Expression('0.0', degree=2)

        # Neumann boundary at x=0
        class B_N(fen.SubDomain):
            def inside(self_, x, on_boundary):
                return on_boundary and fen.near(x[0], 0.0) and x[1]>0.0 and x[1]<self.Ly
            
        # Impedance boundary at x=Lx
        class B_I(fen.SubDomain):
            def inside(self_, x, on_boundary):
                return on_boundary and fen.near(x[0], self.Lx) and x[1]>0.0 and x[1]<self.Ly

        # Dirichlet boundary everywhere else
        class B_D(fen.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and not (B_N().inside(x, 'on_boundary') or B_I().inside(x, 'on_boundary'))

        u_D = fen.Expression('0.0', degree=2)

        TimeHarmonicMaxwellProblem.__init__(self, V, mu, eps, j, B_D(), u_D, B_N(), g_N, B_I(), imp)

    def plot_solution(self, **kwargs):
        """Plot the solution in a 2d colormap"""
        solution = self.get_solution()
        omega = self.get_frequency()
        for i, u in enumerate(solution):
            self.plot_external_solution(u, contains_boundary_values=True, omega=omega[i], **kwargs)

    def plot_solution_trace(self, trace, **kwargs):
        """Plot solution along a trace"""
        solution = self.get_solution(trace=trace)
        for i, u_trace in enumerate(solution):
            plt.figure()
            plt.title('Solution on trace at frequency \u03C9 = {:.3f} rad/s'.format(self.omega[i]))
            all_coords = self.V.tabulate_dof_coordinates()
            trace_coords = np.array([x for x in all_coords if trace.inside(x, 'on_boundary')])
            plt.plot(trace_coords[:, 1], u_trace, **kwargs)
            plt.show()

    def plot_external_solution(self, u_vec, contains_boundary_values=False, omega=None, **kwargs):
        """Plot the external solution u_vec in a 2d colormap"""
        plt.figure()
        if omega is not None:
            plt.title('Solution to system at frequency \u03C9 = {:.3f} rad/s'.format(omega))
        u_func = fen.Function(self.V)
        if not contains_boundary_values:
            u_vec = self.insert_boundary_values(u_vec)
        u_func.vector()[:] = u_vec
        fig = fen.plot(u_func, **kwargs)
        plt.colorbar(fig, orientation='horizontal')
        plt.show()

    def plot_g_N(self, **kwargs):
        """Plot the Neumann boundary condition"""
        plt.figure()
        all_coords = self.V.tabulate_dof_coordinates()
        inlet_coords = np.array([x for x in all_coords if self.B_N.inside(x, 'on_boundary')])
        g_N_coords = [self.g_N(x) for x in inlet_coords]
        plt.plot(inlet_coords[:, 1], g_N_coords, **kwargs)
        plt.ylabel('g_N')
        plt.xlim(0.0, self.Ly)
        plt.show()