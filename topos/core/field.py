import fp
import torch

from topos.io import showField, FieldError

class Field (fp.meta.Functor):
    """
    Scalar functions on the fibers of a domain.

    Fields are internally linked to a torch 1D-tensor
    of the size of the domain.
    """
    
    def __new__(cls, A):
        
        class Field_A (fp.Tens([A.size]), fp.Arrow(fp.Tensor, fp.Tensor)): 
            
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
            def batched(cls, *Ns):
                """ Class for batched fields. """
                domain = fp.Torus([*Ns, int(A.size)])
                FN = cls.functor(domain)
                FN.shape = [*Ns, int(A.size)]
                FN.degree = A.degree
                return FN

            @classmethod
            def batch(cls, xs):
                """ Batch a collection of field instances. """
                if not isinstance(xs, (fp.Tensor, torch.Tensor)):
                    xs = torch.stack([x.data for x in xs])
                Ns = xs.shape[:-1]
                return cls.batched(*Ns)(xs.contiguous())
            
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

            def __call__(self, x): 
                """ Evaluate field on (batched) indices. """
                if not x.data.dtype == torch.long:
                    raise FieldError(f"field arguments should be long indices")
                if x.data.min() < 0 or x.data.max() > self.domain.size:
                    raise FieldError(
                        f"Index range error ({x.data.min(), x.data.max()}) in size {self.domain.size}")
                else: 
                    return self.data.index_select(0, x) 

            def __repr__(self):
                return self.__str__()
                
            def __str__(self):
                return showField(self) 

        name  = cls.name(A)
        bases = (fp.Tens([A.size]), fp.Arrow(fp.Tensor, fp.Tensor))
        dct   = dict(Field_A.__dict__)
        FA = fp.meta.RingMeta(name, bases, dct)
        return FA
    
    @classmethod
    def batched(cls, A, N):
        print("batched")
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