import fp
import torch

from topos.io import showTensor, FieldError

class Field (fp.meta.Functor):
    """
    Scalar functions on the fibers of a domain.

    Fields are internally linked to a torch 1D-tensor
    of the size of the domain.
    """
    
    def __new__(cls, A):
        
        class Field_A (fp.Tens(A.shape)): 

            domain = A
            shape  = [int(A.size)]
            device = A.device if 'device' in dir(A) else 'cpu'
            degree = A.degree if 'degree' in dir(A) else None

            def __new__(cls, data):
                if not isinstance(data, torch.Tensor):
                    data = torch.tensor(data, device=cls.device)
                field = object.__new__(cls)
                field.data   = data
                field.degree = cls.degree
                return field

            def __init__(self, data):
                pass
            
            @classmethod
            def batched(cls, N):
                """ Class for batched fields. """
                domain = fp.Torus([N, int(A.size)])
                FN = cls.functor(domain)
                FN.shape = [N, int(A.size)]
                FN.degree = A.degree
                return FN

            @classmethod
            def batch(cls, xs):
                """ Batch a collection of field instances. """
                N = len(xs)
                if not isinstance(xs, (fp.Tensor, torch.Tensor)):
                    xs = torch.stack([x.data for x in xs])
                return cls.batched(N)(xs)
            
            def items(self):
                """
                Yield (key, value) pairs.
                """
                D = self.domain
                if 'keys' in dir(D):
                    for k, i, j, fiber in zip(D.keys, D.begin, D.end, D.fibers):
                        yield (k, cls(fiber)(self.data[i:j]))
                else:
                    yield(None, D.field(self.data))

            def __getitem__(self, a):
                """
                Local component on the fiber of a.
                """
                K = self.domain
                #---
                if isinstance(a, int) and "grades" not in K.__dir__():
                    return self.data[a]
                #--- Access graded component
                elif isinstance(a, int):
                    begin = sum(K[i].size for i in range(a))
                    end   = begin + K[a].size
                    return K[a].field(self.data[begin:end])
                #--- Access fiber slice
                begin, end, fiber = self.domain.slice(a)
                return cls(fiber)(self.data[begin:end])

            def __setitem__(self, a, va):
                """
                Update a local fiber component.
                """
                begin, end, fiber = self.domain.slice(a)
                try:
                    if isinstance (va, torch.Tensor):
                        data = va.reshape([fiber.size])
                    elif isinstance (va, Field):
                        data = va.data
                    else:
                        data = va * torch.ones([fiber.size])
                    self.data[begin:end] = data
                except:
                    raise FieldError(f"scalar or size {fiber.size} tensor expected")

            def __repr__(self):
                return self.__str__()
                
            def __str__(self):
                if self.domain.size == 0:
                    return ""
                if not 'keys' in dir(self.domain):
                    return showTensor(self.data, 0)
                s = ""
                for k, xk in self.items():
                    fk = self.domain[k]
                    if isinstance(k, torch.LongTensor):
                        sc = f' {k.tolist()} : '
                    else:
                        sc = f' {k} : '
                    pad = len(sc)
                    tensor = (showTensor(xk.data.view(fk.shape), pad) 
                            if "shape" in dir(fk)  else 
                            str(xk).replace("\n", "\n" + " " * pad))
                    s += sc + tensor + "\n"
                return s 

        name  = cls.name(A)
        bases = (fp.Tens([A.size]),)
        dct   = dict(Field_A.__dict__)
        FA = fp.meta.RingMeta(name, bases, dct)
        return FA

    @classmethod
    def batched(cls, A, N):
        domain = fp.Torus([N, int(A.size)])
        domain.__name__ = f'{A.__name__} ({N})'
        BatchedField = cls(domain)
        BatchedField.shape   = [N, int(A.size)]
        BatchedField.batched = True
        return BatchedField

    @classmethod
    def fmap(cls, A):
        pass

    @classmethod
    def name(cls, A):
        name = A.__name__ if '__name__' in dir(A) else "\u03a9"
        return f'Field {name}'
        

class Field2 :
    @classmethod
    def cast2(cls, u, v):
        """
        Return a triple (u', v', c) aimed at a target constructor c.
   
        Check whether v.domain = u.domain.scalars,
        otherwise call Vect.cast2.
        """
        if isinstance(v, u.__class__):
            if u.domain == v.domain:
                return u.data, v.data, u.same
            if u.domain.scalars == v.domain:
                return u.data, u.domain.from_scalars(v).data, u.same
            elif v.domain.scalars == u.domain:
                return v.domain.from_scalars(u).data, v.data, v.same
        return super().cast2(u, v)

    def __init__(self, domain, data=0., degree=None, device=None):
        """
        Create a d-field on a domain from numerical data.

        See Domain.field.
        """
        self.domain = domain
        self.size   = domain.size + (domain.size == 0)
        self.degree = degree
       
        #--- Inputs ---
        if isinstance(data, torch.Tensor):
            self.data = data
        elif isinstance(data, list):
            self.data = torch.tensor(data, device=device)
        elif isinstance(data, (int, float)):
            self.data = data * torch.ones([self.size], device=device)
        else:
            raise FieldError(
                    f"Unsupported data type: {type(data)}",
                    "use torch.Tensor, float, [float]...")
   
        #--- Check shape ---
        if self.data.shape[0] != self.domain.size:
            self.data = self.data.flatten()
        if self.data.shape[0] != self.size:
            raise FieldError(
                f"Could not coerce to domain size {domain.size}",
                f"invalid input shape {list(self.data.shape)}")
        self.device = self.data.device

    def same(self, other=None, name=None):
        """
        Create another field on the same domain.
        """
        #--- Copy ---
        if isinstance(other, type(None)):
            return self.__class__(self.domain, self.data, self.degree)
        #--- Create ---
        if not isinstance (other, Field):
            return self.__class__(self.domain, other, self.degree)
        #--- Pass ---
        if self.domain == other.domain:
            return other
        #--- From scalars ---
        if self.domain.scalars.size == other.domain.size:
            return self.domain.from_scalars(other)

    def norm(self):
        """
        Euclidean norm of the underlying vector.
        """
        return torch.sqrt((self.data.abs() ** 2).sum())

    def sum(self):
        """
        Sum of scalar components.
        """
        return self.data.sum()

    #--- Show ---

    def __repr__(self):
        prefix = self.degree if self.degree != None else ""
        return f"{prefix} Field {self}"

