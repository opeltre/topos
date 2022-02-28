from topos.core     import Shape

class Fiber :

    @classmethod
    def read (cls, key):
        if isinstance(key, cls):
            return key.key
        return key
   
    @classmethod
    def join(cls, keys, shape=None):
        #--- if $1 is a dictionnary holding shapes --
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

    def __init__(self, key:torch.LongTensor, shape=None, begin=0, idx=0):
        """
        Create a fiber from a key, a shape, and optional pointers.

        Attributes:
        -----------
            - key   : identifier of the base point/region (torch.LongTensor)
            - begin : points to start of fiber            (int, default=0)
            - end   : points to the end                   (int)
            - size  : size of fiber shape                 (int, default=1)
            - shape : shape of the fiber                  (Shape, default=Shape())
        """
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

