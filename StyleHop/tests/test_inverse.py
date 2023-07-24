# %%
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from StyleHop import *

A = np.arange(4*4).reshape(1,4,4,1)
arg = {'win':2, 'stride': 2, 'pool': 1, 'pad': 0}
ret = Shrink(A, arg)
recover = invShrink(ret, arg)
if np.sum(A != recover) == 0:
    print("All match")
else:
    print("Not match")

A = np.arange(4*4).reshape(1,4,4,1)
arg = {'win':3, 'stride': 1, 'pool': 1, 'pad': 0}
ret = Shrink(A, arg)
recover = invShrink(ret, arg)
if np.sum(A != recover) == 0:
    print("All match")
else:
    print("Not match")

A = np.arange(3*5).reshape(1,3,5,1)
arg = {'win':3, 'stride': 2, 'pool': 1, 'pad': 0}
ret = Shrink(A, arg)
recover = invShrink(ret, arg)
if np.sum(A != recover) == 0:
    print("All match")
else:
    print("Not match")

A = np.arange(5*5).reshape(1,5,5,1)
arg = {'win':5, 'stride': 1, 'pool': 1, 'pad': 0}
ret = Shrink(A, arg)
recover = invShrink(ret, arg)
if np.sum(A != recover) == 0:
    print("All match")
else:
    print("Not match")

A = np.arange(4*4).reshape(1,4,4,1)
arg = {'win':3, 'stride': 1, 'pool': 1, 'pad': 1}
ret = Shrink(A, arg)
recover = invShrink(ret, arg)
if np.sum(A != recover) == 0:
    print("All match")
else:
    print("Not match")

A = np.arange(7*11).reshape(1,11,7,1)
arg = {'win':5, 'stride': 2, 'pool': 1, 'pad': 2}
ret = Shrink(A, arg)
recover = invShrink(ret, arg)
if np.sum(A != recover) == 0:
    print("All match")
else:
    print("Not match")