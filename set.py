class Set (set):
    """
    Hashable sets as strings.
    """
    def __init__(self, elems=(), sep=':'):
        self.sep = sep 
        if type(elems) == str:
            elems = elems.split(sep) if len(elems) > 0 else []
        super().__init__(elems)
    
    def __hash__(self): 
        return hash(str(self))

    def __str__(self): 
        elems = [str(e) for e in self]
        elems.sort()
        return "(" + self.sep.join(elems) + ")"

    def __repr__(self): 
        return f"Set {str(self)}"

    def __eq__(self, other): 
        return str(self) == str(other)
