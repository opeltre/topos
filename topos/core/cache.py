from .field  import Field
from .linear import Linear
import torch

#--- Cached operators

def linear_cache (name, symbol=None):
    """ 
    Decorator for operator computations.

    Given a method `op : D -> Functional(D, D')`, returns 
    a method with prototypes:
        - `op : D -> Functional(D, D')` to compute or read `op` from cache
        - `op : Field(D) -> Field(D')` applied to arguments. 
    """
    if type(symbol) == type(None): 
        symbol = name

    def decorator(operator):

        def runCached (self, x=None):
            #-- Degree and cache key
            d     = x.degree if isinstance(x, Field) else x
            cache = self._cache if d == None else self[d]._cache
            #-- Read/write cache
            if name in cache:
                op = cache[name]
            else:
                op = operator(self) if d == None else operator(self, d) 
                op.name = symbol
                cache[name] = op
            #-- Apply to x / return op
            if isinstance(x, Field):
                return op @ x
            elif isinstance(x, torch.Tensor):
                return op.tgt.field(op.data @ x)
            return op
        return runCached

    return decorator
