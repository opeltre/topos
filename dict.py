class DictMixin:

    def pull(self, other):
        return self.__class__({x: self[y] for y, x in other})

    def map(self, f): 
        return self.__class__({k: f(v, k) for v, k in self})

    def fmap(self, f): 
        return self.__class__({k: f(v) for v, k in self})

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

    def __str__(self): 
        s = "{\n" if len(self) else "{"
        for v, k in self:
            sv = str(v).replace('\n', '\n    ')
            s += f"{k} :\n {sv}\n\n"
        return s + "}"

    def __repr__(self): 
        return f"Dict {self}"


class Dict (DictMixin, dict):

    def __iter__(self): 
        return ((self[k], k) for k in super().__iter__())

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, val):
        self[key] = val

    def __dir__(self):
        return super().__dir__() + [str(k) for k in self.keys()]


class Record (DictMixin):

    def __init__(self, values={}):

        if isinstance(values, Record):
            values = values.values
        self.values = Dict(values)

    def __contains__(self, key): 
        return key in self.values

    def __iter__(self):
        return self.values.__iter__()
        
    def __getitem__(self, key): 
        return self.values[key]

    def __setitem__(self, key, val):
        self.values.__setitem__(key, val)

    def __len__(self): 
        return len(self.values)
