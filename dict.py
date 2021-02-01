class Dict (dict): 

    def map(self, f): 
        return Dict({k: f(v, k) for v, k in self})

    def map_(self, f):
        for v, k in self:
            self[k] = f(v, k)
        return self

    def filter(self, f):
        out = Dict({})
        for v, k in self:
            if f(v, k):
                out[k] = v
        return out

    def filter_(self, f): 
        keys = []
        for v, k in self: 
            if not f(v, k):
                keys += [k]
        for k in keys:
            self.pop(k)
        return self

    def reduce(self, f, acc): 
        for v, k in self:
            acc = f(acc, v, k)
        return acc

    def __iter__(self): 
        return ((self[k], k) for k in super().__iter__())

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, val):
        self[key] = val

    def __dir__(self):
        return super().__dir__() + [str(k) for k in self.keys()]
   
    def __str__(self): 
        s = "{\n"
        for v, k in self:
            s += f"{k} :\n {v}\n\n"
        return s + "}"

    def __repr__(self): 
        return f"Dict {self}"
