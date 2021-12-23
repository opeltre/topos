import test

from topos import Domain, Sheaf

F = Sheaf({'i': [3], 'j': [3], 'k': [3]})
D = Domain(['i', 'j', 'k'])


class TestSheaf (test.TestCase):

    def test_size(self):
        n, m = F.scalars.size, D.size
        self.assertEqual(n, m)

    def test_sums(self):
        u = F.sums @ F.uniform() 
        v = F.scalars.ones()
        self.assertClose(u, v)

    def test_extend(self):
        u = F.ones()
        v = F.extend @ F.scalars.ones()
        self.assertClose(u, v)

    def test_gibbs(self):
        u = F.gibbs @ F.zeros()
        v = F.uniform()
        self.assertClose(u, v)

        p = F.gibbs(F.randn())
        one = F.scalars.ones()
        self.assertClose(F.sums @ p, one, 1e-6)

    def test_log(self):
        u = F.randn() 
        p = F.gibbs(u)
        v = F._ln(p) 
        self.assertClose(u - F.means(u), v - F.means(v), 1e-6)

    def test_fft(self):
        v = F.fft @ F.ones() 
        u = F.field([1., 0., 0.] * 3)
        self.assertClose(u, v)

    def test_ifft(self):
        u = F.randn()
        self.assertClose(u, F.ifft @ F.fft @ u)

if __name__ == '__main__':
    test.main()
