# Topos

**This library is currently being refactored (sept. 21)**

*Statistical fields are reimplemented as 1D-tensors internally
for efficiency. This means more work to cook index formulas compared
to an implementation leveraging on the natural mathematical structures
involved. On the other hand, all arithmetic operations, map operations
and linear operators benefit from a native parallel implementation 
coming from pytorch.*

## Example: belief network on graphs

See [example.py](example.py)

```
>>> u = K.randn(degree=0)
>>> u
0-Field {

(i:j) ::       [[-1.6016,  0.6941],
                [-0.2367, -0.1504]],

(j:k) ::       [[ 0.3672, -0.0543],
                [ 0.7570, -0.0231]],

(j) ::       [0.5114, 0.5331],

}

>>> d
1 Linear d

>>> d(u)
1-Field {

(i:j) > (j) ::       [ 2.3497, -0.0106],

(j:k) > (j) ::       [ 0.1985, -0.2008],

}
``` 

