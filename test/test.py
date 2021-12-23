import unittest

class TestCase (unittest.TestCase):

    def assertClose (self, u, v, tol=1e-6):
        return self.assertTrue((u - v).norm() < tol)

main = unittest.main
