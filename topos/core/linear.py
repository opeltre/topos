from .vect import Vect
from .field import Field
from .sparse import eye, zero, matmul, diag

from topos.io import LinearError

import torch
import fp

class Linear(fp.Linear): 

    def __new__(cls, A, B):

        Fa, Fb = Field(A), Field(B)
        na, nb = A.size, B.size
        
        class LinAB (fp.Linear([na], [nb]), fp.Arrow(Fa, Fb)):
            
            src = Fa
            tgt = Fb

            def __new__(cls, data, degree=None, name=None):
                lin = object.__new__(cls)
                cls.__init__(lin, data, degree)
                return lin

            @property
            def T(self):
                LinBA = cls(B, A)
                return LinBA(self.data.T)

            def __init__(self, data, degree=None, name=None):
                super().__init__(data)
                self.degree = degree
                if name:
                    self.__name__ = name

            def t(self):
                d, name = self.degree, self.__name__ + '*'
                return Linear(B, A)(self.data.t(), d, name)

            def __truediv__(self, other): 
                """ 
                Divide coefficients by numerical data (e.g. scalar).
                """
                return self.same(self.data / other)

            def __getitem__(self, i):
                if isinstance(i, int):
                    return self.src(self.data[i].to_dense())
                elif isinstance(i, str):
                    return self[self.tgt.index(i)]
        
            def __mul__(self, other):
                if isinstance(other, self.__class__):
                    return super().__mul__(other)
                if isinstance(other, self.src):
                    D = other.domain
                    w = diag(D.size, other.data)
                    out = matmul(self.data, w)
                    return self.__class__(out)
                return super().__mul__(other)

            def __rmul__(self, other):
                if isinstance(other, self.__class__):
                    return super().__rmul__(other)
                if isinstance(other, self.tgt):
                    D = other.domain
                    w = diag(D.size, other.data)
                    out = matmul(w, self.data)
                    return self.__class__(out)
                return super().__rmul__(other)
            
        return LinAB
    
    @classmethod
    def name(cls, A, B):
        rep = lambda D : D.__name__ if '__name__' in dir(D) else D.size
        return f'Linear {rep(A)} -> {rep(B)}'

    @classmethod
    def compose(cls, f, g):
        f_base = fp.Linear(f.src.shape, f.tgt.shape)(f.data)
        g_base = fp.Linear(g.src.shape, g.tgt.shape)(g.data)
        fg = fp.Linear.compose(f_base, g_base)
        return cls(g.src.domain, f.tgt.domain)(fg.data)

    @classmethod
    def source_type(cls, f, xs):
        assert(len(xs) == 1)
        x = xs[0]
        s_x, s_in = tuple(x.shape), tuple(f.src.shape)
        if s_x == s_in: 
            return f.src
        elif s_x[-len(s_in):] == s_in: 
            return f.src.batched(*s_x[:-len(s_in)])

    @classmethod
    def target_type(cls, f, xs):
        x = xs[0]
        s_x = tuple(x.shape)
        s_in, s_out = tuple(f.src.shape), tuple(f.tgt.shape)
        if s_x == s_in:
            return f.tgt
        elif s_x[-len(s_in):] == s_in:
            return f.tgt.batched(*s_x[:-len(s_in)])


class Linear2 (Functional, Vect): 
    """ 
    Linear functionals implemented as sparse matrices.
    """
    
    def __init__(self, domains, mat=0, name="Mat", degree=None):
        """
        Create a Linear operator between domains = [src, tgt].
        """
        src, tgt = domains[0], domains[-1]
        self.degree = degree
        
        #-- Null --
        if isinstance(mat, int) and mat == 0:
            name = "0"
            mat  = zero(tgt.size, src.size)

        #-- Identity --
        if not isinstance(mat, torch.Tensor) and tgt == src:
            name = str(mat)
            mat  = mat * eye(tgt.size)

        #-- Action on fields
        def matvec(x):
            if torch.is_complex(mat) and not torch.is_complex(x.data):
                return tgt.field(mat @ x.data.cfloat())
            return tgt.field(mat @ x.data)

        self.data = mat
        super().__init__([src, tgt], matvec, name)
    
    def __call__(self, field):
        """ Action on fields. """
        tgt, mat = self.tgt, self.data
        # numerical data
        if isinstance(field, Vect):
            x = field.data
        elif isinstance(field, torch.Tensor):
            x = field
        else:
            raise LinearError(f"Invalid input type {type(field)}")
        # matvec product
        if torch.is_complex(mat) and not torch.is_complex(x):
            x = x.cfloat()
        return tgt.field(mat @ x)

    @classmethod
    def null(cls, domains):
        """
        Return the null operator between domains.
        """
        return cls(domains, 0, "0")

    def same(self, data=None, name=None): 
        """
        Return an operator between same domains from a matrix. 
        """
        src, tgt = self.src, self.tgt
        if isinstance(data, type(None)):
            data = self.data
            name = self.name
        elif isinstance(data, torch.Tensor):
            name = name if name != None else self.name 
        return Linear([src, tgt], data, name, self.degree)    

    def __truediv__(self, other): 
        """ 
        Divide coefficients by numerical data (e.g. scalar).
        """
        return self.same(self.data / other)

    def __getitem__(self, i):
        if isinstance(i, int):
            return self.src.field(self.data[i].to_dense())
        elif isinstance(i, str):
            return self[self.tgt.index(i)]
   
    def __mul__(self, other):
        if not isinstance(other, Linear):
            D = other.domain
            w = diag(D.size, other.data)
            if D == self.src:
                out = matmul(self.data, w)
            return self.same(out)
        return super().__mul__(other)

    def __rmul__(self, other):
        if not isinstance(other, Linear):
            D = other.domain
            w = diag(D.size, other.data)
            if D == self.tgt:
                out = matmul(w, self.data)
                return self.same(out)
        return super().__rmul__(other)
        
    def __repr__(self):
        return f"Linear {self}"

    def t(self):
        """ 
        Transposed operator.
        """
        return self.__class__(
            [self.tgt, self.src], 
            self.data.t(),
            name = f"{self}*")


class GradedLinear (GradedFunctional):
    """
    Graded Linear operators.

    An array `L = [L0, ..., Ld]` of linear operators 
    acting on a k-field `f` by:

        L(f) = L @ f = L[k] @ f
        
    """
    
    def __init__(self, domains, mats, degree=0, name="Mat"):
        """
        Create a graded operator from an array of matrices.
        """
        super().__init__(domains, mats, degree, name, cls=Linear)
    
    def null(self, i):
        return Linear([self.src[i], self.tgt[i + self.degree]])

    def t(self): 
        d    = self.degree
        name = f'{self}*'
        ts   = [self.null(i).t() for i in range(d)]\
             + [fi.t() for fi in self.grades]
        return self.__class__([self.tgt, self.src], ts, -d, name) 

    #--- Show ---

    def __repr__(self):
        return f"{self.degree} Linear {self}"
