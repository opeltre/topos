class ToposError (Exception):
    cls = "Topos"
    def __init__(self, m1, m2):
        msg = ("\n" + "-" * 55 + "\n"\
              + f"[{self.cls}]: {m1}\n" + f"\t {m2}\n"\
              + "-" * 55)
        super().__init__(msg)

class VectError (ToposError):
    cls = "Vect"

class FieldError (VectError):
    cls = "Field"

class LinearError (ToposError):
    cls = "Linear"

class IOError (ToposError):
    cls = "IO"