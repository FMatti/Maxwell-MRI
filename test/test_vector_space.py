import unittest
import numpy.testing as testing

import numpy as np
import fenics as fen

import os
os.chdir('/home/fenics/shared/test')

from context import src
from src.vector_space import VectorSpace, VectorSpaceL2
from src.time_harmonic_maxwell_problem import TimeHarmonicMaxwellProblem

class TestCase(unittest.TestCase):
    def setUp(self):
        self.VS = VectorSpace(np.diag(np.ones(10)))

        g_N_symmetric = fen.Expression('sin(x[1]*pi)', degree=2)
        self.THMP_symmetric, B_N = self.get_simple_THMP(g_N_symmetric)
        self.THMP_symmetric.setup()
        self.THMP_symmetric.solve([1])
        g_N_asymmetric = fen.Expression('sin(x[1]*2*pi)', degree=2)
        self.THMP_asymmetric, B_N = self.get_simple_THMP(g_N_asymmetric)
        self.THMP_asymmetric.setup()
        self.THMP_asymmetric.solve([1])
        self.VSL2_trace = VectorSpaceL2(self.THMP_asymmetric, trace=B_N())

    def get_simple_THMP(self, g_N):
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

        return TimeHarmonicMaxwellProblem(V, mu, eps, j, B_D(), u_D, B_N(), g_N), B_N

    def test_norm(self):
        x = np.ones(10)
        testing.assert_almost_equal(self.VS.norm(x),
                                    pow(10, 0.5),
                                    err_msg='incorrect norm')

    def test_complex_norm(self):
        x = np.ones(10) + 1j*np.ones(10)
        testing.assert_almost_equal(self.VS.norm(x),
                                    pow(20, 0.5),
                                    err_msg='incorrect complex norm')

    def test_inner_product(self):
        x = np.ones(10)
        y = np.resize([1, -1], 10)
        testing.assert_almost_equal(self.VS.inner_product(x, y),
                                    0,
                                    err_msg='incorrect inner product')

    def test_matrix_inner_product(self):
        x = np.ones((2, 10))
        y = np.c_[np.resize([1, -1], 10), np.resize([-1, 1], 10)].T
        testing.assert_almost_equal(self.VS.inner_product(x, y),
                                    np.zeros((2, 2)),
                                    err_msg='incorrect matrix inner product')


    def test_inner_product_THMP_trace(self):
        x = self.THMP_symmetric.get_solution(tonumpy=True, trace=self.VSL2_trace.get_trace())
        y = self.THMP_asymmetric.get_solution(tonumpy=True, trace=self.VSL2_trace.get_trace())
        testing.assert_almost_equal(self.VSL2_trace.inner_product(x, y),
                                    0,
                                    err_msg='incorrect trace inner product')

def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestCase('test_norm'))
    suite.addTest(TestCase('test_complex_norm'))
    suite.addTest(TestCase('test_inner_product'))
    suite.addTest(TestCase('test_matrix_inner_product'))
    suite.addTest(TestCase('test_inner_product_THMP_trace'))
    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
