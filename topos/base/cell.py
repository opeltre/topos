from .hashable import Hashable 

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

def join_cells(keys, shape): 
    cells = {}
    begin = 0
    for i, k in enumerate(keys):
        cell        = Cell(k, i, shape[k], begin = begin)
        cells[k]    = cell
        begin       += cell.size
    return cells, begin

