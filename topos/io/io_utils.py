import torch

def alignString (alinea='', s='', prefix='tensor(', suffix=')'):
    return alinea + (str(s).replace(prefix, ' ' * len(prefix))
                           .replace(suffix, '')
                           .replace('\n', '\n' + ' ' * len(alinea)))

def readTensor(js):
    return torch.tensor(js) if not isinstance(js, torch.Tensor) else js
