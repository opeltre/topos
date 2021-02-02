from system import System 
from field import Field

K = System(('i:j:k', 'i:k:l', 'j:k:l'))

f = Field(K, 0, "gaussian") 
phi = Field(K, 1, "gaussian")
