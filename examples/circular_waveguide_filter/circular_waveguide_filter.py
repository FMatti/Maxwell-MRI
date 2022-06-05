# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

import fenics as fen

from context import src
from src.time_harmonic_maxwell_problem import TimeHarmonicMaxwellProblem

class CircularWaveguideFilter(TimeHarmonicMaxwellProblem):
    """
    Daughter class of TimeHarmonicMaxwellProblem specialized on 3d waveguides.

    Members
    -------
    (Additional members inherited from TimeHarmonicMaxwellProblem)

    Methods
    -------
    plot_solution() : None -> None
        Plot the solution as a 3d vector field.
    (Additional methods inherited from TimeHarmonicMaxwellProblem)
    """

    def __init__(self, mesh, B_N, g_N):
        # Import mesh from specified filepath and scale its units to meters
        mesh = fen.Mesh(mesh)
        mesh.coordinates()[:] = mesh.coordinates()[:] * 1e-3

        # Use Nédélec elements of the first kind
        V = fen.FunctionSpace(mesh, 'N1curl', 1)

        # Physical constants in vacuum
        mu = fen.Expression('4e-7*pi', degree=2)
        eps = fen.Expression('8.854187e-12', degree=2)
        j = fen.Expression(('0.0', '0.0', '0.0'), degree=2)

        # Dirichlet boundary on all non-Neumann boundaries
        class B_D(fen.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and not B_N().inside(x, 'on_boundary')

        # Perfectly conducting boundary condition
        u_D = fen.Expression(('0.0', '0.0', '0.0'), degree=2)

        TimeHarmonicMaxwellProblem.__init__(self, V, mu, eps, j, B_D(), u_D, B_N(), g_N)

    def plot_solution(self):
        """Plot the solution as a vector field"""
        u_sol = self.get_solution()
        for i, u in enumerate(u_sol):
            plt.figure()
            plt.title(f'Solution to system at frequency \u03C9 = {self.omega[i]} rad/s')
            fig = fen.plot(u)
            plt.show()
