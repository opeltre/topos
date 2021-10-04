from .hashable  import Hashable 
from .shape     import Shape

class Fiber (Hashable): 

    @staticmethod
    def join(keys, shape):
        fibers = {}
        begin = 0
        for i, k in enumerate(keys):
            fiber        = Fiber(k, i, shape[k], begin = begin)
            fibers[k]    = fiber
            begin       += fiber.size
        return fibers, begin

    def __init__(self, key, i=0, shape=None, begin=0):
        if not isinstance(shape, Shape):
            shape   = Shape(*(shape if shape else []))
        self.key    = key 
        self.idx    = i
        self.begin  = begin
        self.end    = begin + shape.size
        self.shape  = shape
        self.size   = shape.size

    def to (begin):
        self.begin = begin
        self.end   = begin + self.size

    def __gt__(self, other): 
        return self.key > other.key

    def __ge__(self, other): 
        return self.key >= other.key

    def __str__(self): 
        return str(self.key)
    
    def __repr__(self): 
        return f"Fiber {self} {self.begin}-{self.end}"
