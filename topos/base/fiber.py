from .hashable      import Hashable 
from .shape         import Shape
from .set           import Set

class Fiber (Hashable): 

    @classmethod
    def read (cls, key):
        if isinstance(key, cls):
            return key.key
        return key

    @classmethod
    def join(cls, keys, shape=None):
        #--- dictionnary of shapes in $1 --
        if isinstance(keys, dict) and shape == None:
            shape = {cls.read(k): Ek for k, Ek in shape.items()}
            keys  = shape.keys()
        else:
            keys  = [cls.read(k) for k in keys]
            #--- pointwise shapes ---
            if shape == None:
                shape = {k : Shape() for k in keys}
            #--- shapes from callable ---
            elif callable(shape):
                shape = {k : shape(k) for k in keys}\
        #--- join ---
        fibers = {}
        begin = 0
        for i, k in enumerate(keys):
            fiber        = Fiber(k, shape[k], begin, i)
            fibers[k]    = fiber
            begin       += fiber.size
        return fibers, begin

    def __init__(self, key, shape=None, begin=0, idx=0):
        if isinstance(shape, type(None)):
            shape = Shape()
        elif not isinstance(shape, Shape):
            shape   = Shape(*shape) if len(shape) > 0 else Shape()
        self.key    = key 
        self.idx    = idx
        self.begin  = begin
        self.end    = begin + shape.size
        self.shape  = shape
        self.size   = shape.size

    def __gt__(self, other): 
        return self.key > other.key

    def __ge__(self, other): 
        return self.key >= other.key

    def __str__(self): 
        return str(self.key)
    
    def __repr__(self): 
        return f"Fiber {self} {self.begin}-{self.end}"

