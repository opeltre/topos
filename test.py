from system import System 
from field import Field

K = System(('i:j:k', 'i:k:l', 'j:k:l')).close()

f = Field(K, 0) 
phi = Field(K, 1)
