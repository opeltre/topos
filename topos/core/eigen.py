from .linear import Linear

import fp
import torch

def loop(n): 
    def decorator(f):
        def looped(x):
            for k in range(n):
                x = f(x)
            return x
        return looped
    return decorator
                

class Eigen:
    """
    Find eigenspaces and eigenvalues of a real hermitian operator. 

    Projectively exponentiating a matrix `A` on the operator-norm 
    sphere `||A|| = 1`, i.e. taking the limit:  
    
        P = lim_n  A^n / ||A^n|| 

    yields the orthogonal projector onto the dominant eigenspace of A.
    """

    @staticmethod    
    def exp_norm(log2N):
        """ 
        Projective exponentiation on the sphere of linear operators.

        Returns `exp(2**n): Linear(E, E) -> Linear(E, E).`

        When the input `A` is symmetric, converges to the projector on 
        the dominant eigenspace of `A`. 
        """
        @loop(log2N)
        def run(A):
            A2 = A @ A
            if not A2.is_floating_point():
                A2 = A2.float()
            n2 = A2.norm()
            return A2 * (1/n2)
        return run

    @staticmethod
    def one_hot(i, n, device=None):
        """ 
        Canonical basis vector `ei` of size n. 
        """
        ei = torch.zeros(n, dtype=torch.float, device=device)
        ei[i] = 1
        return ei 

    @staticmethod
    def normalize(x):
        return x / x.norm()

    @classmethod
    def split(cls, A, log2N=32):
        """ 
        Return an orthogonal splitting `(vec, val, B)`. 

        The returned operator `B` acts on the orthogonal supplement 
        of the eigenvector `vec`, with eigenvalue `val`.
        """
        n = A.shape[0]
        #--- dominant projector ---
        P = cls.exp_norm(log2N)(A)
        #--- dominant eigenvector
        i = torch.max(A.norm(dim=[0]), 0).indices
        ei = cls.one_hot(i, n)
        xi = cls.normalize(P @ ei)
        #--- dominant eigenvalue
        vi = (A @ xi).norm()
        return xi, vi, A - vi * P

    @classmethod
    def elements(cls, A, rank=None, log2N=32):
        """
        Return dominant eigenvectors and eigenvalues. 
        """
        n = A.shape[0]
        rank = n if type(rank) == type(None) else rank
        vecs, vals, B = [], [], A
        for k in range(rank):
            xi, vi, B = cls.split(B, log2N)
            vecs.append(xi)
            vals.append(vi)
        return torch.stack(vecs), torch.stack(vals)