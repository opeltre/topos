from .io_utils import (
    readKey, readFunctor, readTensor, 
    showTensor, alignString, print_options)

from .exceptions import ToposError, FieldError, VectError, LinearError

from .plot import plot_graph, plot_contours, plt