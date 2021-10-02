class Graded: 

    def __getitem__(self, i):
        return self.grades[i]

    def __iter__(self): 
        return self.grades.__iter__()

    #--- Arithmetic ---

    def __neg__(self):
        return self.same([-xi for xi in self])

    def __add__(self, other):
        return self.same([xi + yi for xi, yi in zip(self, other)])

    def __sub__(self, other):
        return self.same([xi - yi for xi, yi in zip(self, other)])

    def __mul__(self, other):
        return self.same([xi * yi for xi, yi in zip(self, other)])

    def __truediv__(self, other):
        return self.same([xi / yi for xi, yi in zip(self, other)])

    def __repr__(self):
        return f"{self.degree} {super().__repr__()}"
