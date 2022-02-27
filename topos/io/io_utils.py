import torch

#--- Region keys 

def readKey(k):
    if isinstance(k, (int, str)):
        return k
    return tuple(readKey(ki) for ki in k)

#--- Tensor I/O

def readTensor(js):
    return torch.tensor(js) if not isinstance(js, torch.Tensor) else js

def showTensor (t, pad):
    return str(t).replace("tensor(", " " * 7)\
        .replace(")", "")\
        .replace("\n\n", "\n")\
        .replace("\t", "")\
        .replace(r'\s*', "")\
        .replace("\n", "\n" + " " * pad)

def alignString (alinea='', s='', prefix='tensor(', suffix=')'):
    return alinea + (str(s).replace(prefix, ' ' * len(prefix))
                           .replace(suffix, '')
                           .replace('\n', '\n' + ' ' * len(alinea)))

