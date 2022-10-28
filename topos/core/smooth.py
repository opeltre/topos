from re import X
import fp

from .field  import Field
from .linear import Linear

class Smooth (fp.Arrow):
    
    def __new__(cls, A, B):

        Fa, Fb = Field(A), Field(B)

        class SmoothAB (fp.Arrow(Fa, Fb)):

            src = Fa
            tgt = Fb

            def batch(self, N):
                Smooth = self.functor
                return Smooth.batched(A, B, N)(self)

            def __init__(self, f, df=None, name='\u03bb'):
                super().__init__(f)
                self.__name__ = name
                if callable(df):
                    self.tangent = fp.Arrow(Field(A), Linear(A, B))
            
        return SmoothAB
    
    @classmethod
    def batched(cls, A, B, N):
        """ Class for batched maps. """
        SmoothAB = cls(A, B)
        FA = Field.batched(A, N)
        FB = Field.batched(B, N)

        class BatchedSmooth(SmoothAB, fp.Arrow(FA, FB)):

            src = FA
            tgt = FB
            batched = True

        BatchedSmooth.__name__ = SmoothAB.__name__ + f' ({N})'
        return BatchedSmooth

    @classmethod
    def compose(cls, f, g):
        return SmoothPipe(g.src.domain, f.tgt.domain)([g, f])

    @classmethod
    def name(cls, A, B):
        rep = lambda D: D.__name__ if '__name__' in dir(D) else D.size
        return f'Smooth {rep(A)} -> {rep(B)}'

class SmoothPipe(Smooth):

    def __new__(cls, A, B):

        class Pipe_AB(Smooth(A, B)):
            
            def __init__(self, fs, name='\u03bbs'):
                self.__name__ = name
                self.pipe = fs
                def call(x):
                    out = x
                    for f in fs:
                        out = f(out)
                    return out
                super().__init__(call, None, name)
                self.call = call

            def batch(self, N):
                Fs = [f.batch(N) for f in self.pipe]
                name = self.__name__
                Cls = self.functor.batched(A, B, N)
                return Cls(Fs, name)                
        
        return Pipe_AB


class VectorField:

    def __new__(cls, A):

        class VecA(Smooth(A, A)):
            
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._writers = []
            
            @cls.integrator
            def euler(self, dt):
                """ Explicit Euler integrator. """
                return lambda x: x + dt * self(x)

            def writer(self, callback):
                """ 
                Use as decorator to append an iteration callback.

                The writer will be called as `callback(x, n)` where:
                    - `x` is the vector field argument,
                    - `n` is the current iteration number. 
                """
                self._writers.append(callback)
                return callback
        
        VecA.__name__ = f'VectorField {A}'
        return VecA

    @classmethod   
    def integrator(cls, method):

        def loop_method(self, dt, n=1):

            step = method(self, dt)

            @fp.Arrow(self.src, self.tgt)
            def integrate(x0):
                x = x0 + 0
                for k in range(n):
                    #--- writers ---
                    if len(self._writers):
                        for w in self._writers:
                            w(x, n)
                    #--- callback ---
                    x = step(x)
                return x
            integrate.__name__ = f'{method.__name__} {dt} x {n}'
            return integrate

        return loop_method