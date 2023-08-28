import topos
import torch
import torch.nn as nn 

class GCN(nn.Module): 
     """ Glass Convolutional Network with scalar channel. """

     def __init__(self, G, degree=2, Cout=1):
          super().__init__()
          self.src = topos.Complex(G)
          self.Cin = self.src.size // self.src.scalars.size
          self.Cout = Cout
          self.weights = nn.Parameter(torch.randn(Cout, (1 + degree) * self.Cin))
          #--- Powers of the graph laplacian --- 
          L = self.graph.laplacian(0)
          Lpow = [self.graph.eye(0)]
          for i in range(degree):
               Lpow.append(L @ Lpow[-1])
          self.Lpow = [Lp.data for Lp in Lpow]

     def filter(self):
          """ 
          Linear convolutional filter, polynomial of the laplacian. 
          """                  
          G = self.graph
          W = G.eye(0)
          #--- Convolutional filters are polynomials of the laplacian --- 
          for wk, Lk in zip(self.weights, self.Lpow):
               W = W + wk * Lk
          W.__name__ = 'GCN'
          return W

     def forward(self, x):
          #--- Cast to topos.Field instance
          if isinstance(x, torch.Tensor):
               x = self.graph[0].field(x)
          return self.filter() @ x