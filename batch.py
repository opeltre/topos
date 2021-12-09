import topos
import torch

from topos.core import sparse

class Batch (topos.Sum): 

    def __init__(self, N, domain):
        self.nbatch = N
        self.trivial = domain.trivial
        self.scalars = (self.__class__(N, domain.scalars) 
                        if not self.trivial else self)
        super().__init__(*(domain for n in range(N)))

    @staticmethod
    def avg (x):
        N   = x.domain.nbatch
        tgt = x.domain[0]
        return tgt.field(x.data.view([N, -1]).mean([0]))
    
    @staticmethod
    def select (js, x):
        N = x.domain.nbatch
        if isinstance(js, int):
            js = torch.randint(N, [js])
        P = len(js)
        tgt = Batch(P, x.domain[0])
        return tgt.field(x.data.view([N, -1]).index_select(0, js))

    @staticmethod
    def burn (n, x):
        N = x.domain.nbatch
        tgt = Batch(N - n, x.domain[0])
        return tgt.field(x.data.view([N, -1])[n:])
    
    @classmethod
    def lift (cls, N, f):
        src, tgt = cls(N, f.src), cls(N, f.tgt)

        if isinstance(f, topos.Linear):
            data = f.data.coalesce()
            idx  = data.indices().t()
            val  = data.values()

            a, b = f.src.size, f.tgt.size
            ba   = torch.tensor([b, a])[None,:]
            indices = torch.cat([idx + n * ba for n in range(N)])
            values  = torch.cat([val for n in range(N)])
            mat  = sparse.matrix([N * b, N * a], indices, values)

            return topos.Linear([src, tgt], mat, f"[{f.name}]")

        if isinstance(f, topos.Functional):
            def liftf (xs):
                ys = torch.cat([
                    f(f.src.field(x)).data for x in xs.data.view([N, -1])
                ])
                return tgt.field(ys)
            return topos.Functional([src, tgt], liftf, f'[{f.name}]')
