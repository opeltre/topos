from base.hashable import Hashable

class Shape :

    def __init__(self, *ns):
        self.dim = len(ns)
        self.n = list(ns)
        size = 1
        for ni in ns:
            size *= ni
        self.size = size

    def index(self, *js):
        i = js[0]
        for d in range(1, min(len(js), self.dim)):
            i = (i * self.n[d]) + js[d]
        return i

    def __iter__(self):
        return self.n.__iter__()
    
    def __str__(self): 
        return "(" + ",".join([str(ni) for ni in self.n]) + ")"

    def __repr__(self): 
        return f"Shape {self}"

class Cell (Hashable): 

    def __init__(self, key, i, shape, begin=0):
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
        return f"Cell {self} {self.begin}-{self.end}"
