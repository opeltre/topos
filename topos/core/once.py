def once(method):
    def method_once(self, *args, **kwargs):
        name = '_' + method.__name__ 
        if name in dir(self):
            return getattr(self, name)
        out = method(self, *args, **kwargs)
        setattr(self, name, out)
        return out
    return method_once