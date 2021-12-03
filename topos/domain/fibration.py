from .cartesian import Sum

class Fibration (Sheaf, Sum):

    def __init__(self, mapping)
        self.grades = mapping
        self.rank = len(mapping) - 1
        self.trivial = \
            not sum((not f.trivial for f in mapping.values()))
        if "scalars" not in self.__dir__() and not self.trivial:
            self.scalars =\
                self.__class__(*(f.scalars for f in mapping.values))

        #--- Join fibers ---
        shape = {
            Sequence.read((str(i), a)): fa.shape \
                                        for i, Fi in enumerate(sheaves) \
                                        for a, fa in Fi.fibers.items()} 
        keys = shape.keys()
        shape = shape if not self.trivial else None
        super().__init__(keys, shape, ftype=Sequence)
