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

            def __new__(cls, f, df=None, name='\u03bb'):
                map = object.__new__(cls)
                cls.__init__(map, f, df, name)
                return map

            def __init__(self, f, df=None, name='\u03bb'):
                super().__init__(f)
                self.__name__ = name
                if callable(df):
                    self.tangent = fp.Arrow(Field(A), Linear(A, B))

        return SmoothAB
    
    @classmethod
    def name(cls, A, B):
        rep = lambda D: D.__name__ if '__name__' in dir(D) else D.size
        return f'Smooth {rep(A)} -> {rep(B)}'


def integrator(method):
    def loop_method(self, dt, n=1):
        step = method(dt)
        @fp.Arrow(self.src, self.tgt)
        def integrate(x0):
            x = x0 + 0
            for k in range(n):
                x = step(x)
            return X
        return integrate
    return loop_method


class VectorField:

    def __new__(cls, A):

        class VecA(Smooth(A, A)):

            @cls.integrator
            def euler(self, dt):
                """ Explicit Euler integrator. """
                return lambda x: x + dt * self(x)
        
        VecA.__name__ = f'VectorField {A}'
        return VecA

    @classmethod   
    def integrator(cls, method):

        def loop_method(self, dt, n=1):

            step = cls.runStep(method(self, dt))

            @fp.Arrow(self.src, self.tgt)
            def integrate(x0):
                x = x0 + 0
                for k in range(n):
                    x = step(x)
                return x
            integrate.__name__ = f'{method.__name__} {dt} x {n}'
            return integrate

        return loop_method

    @classmethod
    def runStep(cls, step):
        return step


class Diffusion(VectorField):

    beta = 1

    @classmethod
    def runStep(cls, step):
        N = cls.src
        F = N.from_scalars() @ N.freeEnergy(cls.beta)
        retract = lambda H : H - F(H)
        return lambda H:retract(step(H))