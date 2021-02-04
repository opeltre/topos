from set import Set

class Simplex (Set): 
    """
    Hashable lists as strings
    """
    def __init__(self, vertices, sep=">"):
        self.vertices = vertices.split(sep) \
                      if type(vertices) == str else vertices
        self.dim = len(self.vertices) - 1
        super().__init__(vertices, sep=sep)
    
    def __getitem__(self, i): 
        return self.vertices[i]

    def face(self, i): 
        return Simplex(self[:i] + self[i+1:])
    
    def faces(self): 
        if self.dim == 0:
            return []
        return [self.face(i) for i in range(0, self.dim + 1)]

    def __str__(self): 
        elems = [str(e) for e in self]
        return "(" + self.sep.join(elems) + ")" 

    def __repr__(self):
        return f"{self.dim}-Simplex {self}"

    def __iter__(self): 
        return (v for v in self.vertices)
