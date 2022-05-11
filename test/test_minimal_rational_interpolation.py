import unittest
import numpy.testing as testing

import fenics as fen
import os
os.chdir('/home/fenics/shared/test')

from context import src
from src.minimal_rational_interpolation import MinimalRationalInterpolation
from src.vector_space import VectorSpaceL2
from src.time_harmonic_maxwell_problem import TimeHarmonicMaxwellProblem

class TestCase(unittest.TestCase):
    def setUp(self):
        self.THMP, B_N = self.get_simple_THMP()
        self.THMP.setup()
        self.THMP.solve([1, 2, 3, 4, 5])
        self.VS = VectorSpaceL2(self.THMP)
        self.MRI = MinimalRationalInterpolation(self.VS)
        self.VS_trace = VectorSpaceL2(self.THMP, trace=B_N())
        self.MRI_trace = MinimalRationalInterpolation(self.VS_trace)

    def get_simple_THMP(self):
        mesh = fen.UnitSquareMesh(10, 10, 'crossed')
        V = fen.FunctionSpace(mesh, 'P', 1)

        mu = fen.Expression('1.0', degree=2)
        eps = fen.Expression('1.0', degree=2)
        j = fen.Expression('0.0', degree=2)

        class B_N(fen.SubDomain):
            def inside(self_, x, on_boundary):
                return on_boundary and fen.near(x[0], 0.0) and x[1]>0.0 and x[1]<1.0

        class B_D(fen.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and not B_N().inside(x, 'on_boundary')

        u_D = fen.Expression('0.0', degree=2)
        g_N = fen.Expression('sin(x[1]*pi)', degree=2)

        return TimeHarmonicMaxwellProblem(V, mu, eps, j, B_D(), u_D, B_N(), g_N), B_N

    def test_householder(self):
        snapshots = self.THMP.get_solution(tonumpy=True)
        self.MRI._householder_triangularization(snapshots)
        testing.assert_almost_equal(self.MRI.R.T @ self.MRI.R,
                                    self.VS.inner_product(snapshots, snapshots),
                                    err_msg='incorrect root')

    def test_householder_trace(self):
        snapshots = self.THMP.get_solution(tonumpy=True, trace=self.VS_trace.get_trace())
        self.MRI_trace._householder_triangularization(snapshots)
        testing.assert_almost_equal(self.MRI_trace.R.T @ self.MRI_trace.R,
                                    self.VS_trace.inner_product(snapshots, snapshots),
                                    err_msg='incorrect root')

def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestCase('test_householder'))
    suite.addTest(TestCase('test_householder_trace'))
    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
