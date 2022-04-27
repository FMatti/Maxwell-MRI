import unittest
import numpy.testing as testing

import os
os.chdir('/home/fenics/shared/test')

from context import src
from src.rational_function import RationalFunction

class TestCase(unittest.TestCase):
    def setUp(self):
        self.RF = RationalFunction([0, 1], [1, 1], [1, 2])

    def test_evaluate(self):
        testing.assert_almost_equal(self.RF.evaluate(-1.),
                                    4/3,
                                    err_msg='incorrect function evaluation')

    def test_evaluate_pole(self):
        testing.assert_almost_equal(self.RF.evaluate(0.),
                                    1.0,
                                    err_msg='incorrect function evaluation')
    
    def test_evaluate_array(self):
        testing.assert_almost_equal(self.RF.evaluate([2., 3.]),
                                    [5/3, 1.6],
                                    err_msg='incorrect function evaluation')

    def test_evaluate_array_pole(self):
        testing.assert_almost_equal(self.RF.evaluate([1., 2.]),
                                    [2, 5/3],
                                    err_msg='incorrect function evaluation')

    def test_root(self):
        testing.assert_almost_equal(self.RF.roots(),
                                    [0.5 + 0.j],
                                    err_msg='incorrect root')

    def test_get_denominator_argmin(self):
        testing.assert_almost_equal(self.RF.get_denominator_argmin([2., 3., 4.]),
                                    2,
                                    err_msg='incorrect root')

def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestCase('test_evaluate'))
    suite.addTest(TestCase('test_evaluate_pole'))
    suite.addTest(TestCase('test_evaluate_array'))
    suite.addTest(TestCase('test_evaluate_array_pole'))
    suite.addTest(TestCase('test_root'))
    suite.addTest(TestCase('test_get_denominator_argmin'))
    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
