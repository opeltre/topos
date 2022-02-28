from .field import Field

#--- Cached operators

def linear_cache (operator):
    """ 
    Decorator for operator computations.

    Given a method `op : D -> Functional(D, D')`, returns 
    a method with prototypes:
        - `op : D -> Functional(D, D')` to compute or read `op` from cache
        - `op : Field(D) -> Field(D')` applied to arguments. 
    """
    def runCached (self, x=None):
        #-- Degree and cache key
        d    = x.degree if isinstance(x, Field) else x
        name = operator.__name__ \
             + (f':{d}' if d != None else '')
        #-- Read cache
        if name in self._cache:
            op = self._cache[name]
            return op @ x if isinstance(x, Field) else op
        #-- Compute and write cache
        op = operator(self) if d == None else operator(self, d) 
        self._cache[name] = op
        return op @ x if isinstance(x, Field) else op

    return runCached
