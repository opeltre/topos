import torch

from fp.meta import Functor

from .exceptions import IOError

#--- Region keys

def readKey(k):
    if isinstance(k, (int, str)):
        return k
    return tuple(readKey(ki) for ki in k)

#--- Functor I/O

def readFunctor(keys=None, functor=None):
    """
    Parse functorial data.

    Inputs:
        - keys    : source domain (e.g. list)
        - functor : target object shapes (e.g. list)

    Output:
        - idx     : index map from keys to integers
        - keys    : source domain
        - fibers  : target objects
    """
    #--- Dict input ---

    #(functor={'a' : [3], 'b' : [2, 2], ...})
    if isinstance(functor, dict):
        i, idx, keys, fibers = 0, {}, [], []
        for k, fk in functor.items():
            idx[k] = i
            i += 1
            keys   += [k]
            fibers += [fk]
        return idx, keys, fibers
    #({'a': [3], 'b': [2, 2], ...})
    if isinstance(keys, dict):
        return readFunctor(functor=keys)

    #--- List input

    #(functor=[[3], [2, 2], ...])
    if isinstance(functor, list) and type(keys) == type(None):
        keys = list(range(len(functor)))
    #([[3], [2, 2], ...])
    if isinstance(keys, list):
        if not all(isinstance(k, (tuple, int, str)) for k in keys):
            return readFunctor(functor=keys)

    #--- Adjacency matrix input ---

    #(keys:torch.sparse_coo_tensor)
    if isinstance(keys, torch.Tensor):
        shape = keys.shape
        ij    = keys.coalesce().indices()
        keys  = ij.T
        N     = keys.shape[0]
        val = torch.arange(N)
        idx = torch.sparse_coo_tensor(ij, val, size=shape, dtype=torch.long)
        idx = idx.coalesce()

    #--- Key value pairs ---

    #(['a', 'b', ...], [[3], [2, 2], ...])
    elif isinstance(keys, list):
        keys  = keys
        idx   = {readKey(k): i for i, k in enumerate(keys)}

    elif 'size' in dir(keys):
        keys = torch.arange(keys.size)
        idx  = keys

    #--- Functor values
    if isinstance(functor, list):
        fibers = functor
    if isinstance(functor, type(None)):
        fibers = [None for k in keys]
    if callable(functor):
        fibers = [functor(k) for k in keys]

    #--- Return
    if isinstance(fibers, list):
        return idx, keys, fibers

    raise IOError('readFunctor: Invalid key, functor input')


#--- Tensor I/O

def readTensor(x, dtype=None, device=None):
    return (x if isinstance(x, torch.Tensor)
              else torch.tensor(x, dtype=dtype, device=device))

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