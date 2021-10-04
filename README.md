# Topos

[hypergraph]: https://en.wikipedia.org/wiki/Hypergraph
[nerve]: https://en.wikipedia.org/wiki/Nerve_(category_theory)
[divergence]: https://en.wikipedia.org/wiki/Divergence

This library implements topological and statistical structures 
considered in [[2]][phd] for the
study of message-passing algorithms. 

There exists natural boundary operators on statistical systems, 
generalising gradient and [divergence], leading to diffusion equations 
that improve belief propagation (BP) algorithms. 

BP algorithms were originally introduced to efficiently 
estimate the marginals of a high dimensional probability distribution 
(usually performing better than Monte Carlo methods), 
but our geometrical picture leads to new algorithms with even
more interesting applications. 


## Installation 

The only requirement should be pytorch: `pip install torch`.

Then clone the repository locally to start using the library. 

```
git clone https://github.com/opeltre/topos
```

## Usage 

### Systems 

![nerve](assets/img/nerve.png)

System instances describe collections of variables (vertices) 
along with their allowed joint measurements (cells or regions). 
Each variable is assumed binary (2 states) by default.

A system can for example be a graph (all regions are pairs) 
or a higher dimensional instance, called a [hypergraph][hypergraph]. 
For generalized belief propagation (GBP) to run well,
the underlying hypergraph should be closed under intersection. 

```py
from topos import System
K = System.closure(("i:j:k", "j:k:l", "i:k:l"))
```

When a `System` instance is created, all inclusion relations 
are computed to yield the [nerve][nerve] of the hypergraph. 

A collection of topological and combinatorial operators 
are also computed: they act on Field instances. 

### Fields 

![fields](assets/img/fields.png)

A 0-Field `u` is a collection of tensors indexed over regions 
such that `u[a]` is joint observable on variables in `a`.

```py
>>> u = K.zeros(0) 
>>> u["j:k"]
torch.tensor([[.0, .0],
              [.0, .0]])
```

A 1-Field `phi` is a collection of tensors indexed by 1-chains 
such that `phi[a > b]` is a function on the state of variables in `b`. 

```py
>>> phi = K.randn(1)
>>> phi["j:k > k"]
torch.tensor([-0.5114, 0.5331])
```

### Operators  

There is a collection of natural operators acting on such
statistical systems, revealing a rich interplay with topology 
and combinatorics. 

These include (implemented):
- (+1)-graded differential operator `K.d`
- (-1)-graded codifferential operator `K.delta`
- graded zeta transform `K.zeta` 
- graded Möbius transform `K.mu` inverting `K.zeta`
- Gibbs state map `K.gibbs` mapping observables to local probabilities
- the effective energy gradient `K.Deff` from 0-fields to 1-fields
- its tangent map `K.nabla(p)` at a consistent belief `p`. 

See [[1]](#ref1)  and [[2]](#ref2) for a better description of 
these operators and their role in the design of
belief propagation algorithms. 

(In particular, have a look at [algorithms][alg_table]
and [notation table][not_table])

## Example: belief network on graphs

See [example.py](example.py)

```py
>>> K = System(("i:j", "j:k"))

>>> u = K.randn(0)
>>> u
0 Field {

(i:j) ::       [[-1.6016,  0.6941],
                [-0.2367, -0.1504]],

(j:k) ::       [[ 0.3672, -0.0543],
                [ 0.7570, -0.0231]],

(j) ::       [0.5114, 0.5331],

}

>>> K.d
1 Linear d

>>> K.d(u)
1 Field {

(i:j) > (j) ::       [ 2.3497, -0.0106],

(j:k) > (j) ::       [ 0.1985, -0.2008],

}
``` 

# References 

<span id="ref1"></span>
[1] : Peltre - _Belief Propagation as Diffusion_ (2021).
GSI'21 proceedings, [arXiv:2107.12230][gsi21]

<span id="ref2"></span>
[2] : Peltre - _Message-Passing Algorithms and Homology_ (2020).
PhD preprint, [arXiv:2009.11631][phd]

<span id="ref3"></span>
[3] : Yedidia, Freeman, Weiss - _Generalized Belief Propagation_ (2000).
NeurIPS 2000, [full text][YFW00]

[gsi21]: https://arxiv.org/abs/2107.12230
[phd]:   https://arxiv.org/abs/2009.11631
[YFW00]: https://https://proceedings.neurips.cc/paper/1832-generalized-belief-propagation.pdf
[not_table]: https://arxiv.org/pdf/2009.11631#page=4
[alg_table]: https://arxiv.org/pdf/2107.12230#page=7
