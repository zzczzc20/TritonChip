import sys, random
sys.path.insert(0, "/home/zhongchun/r42/kfuse/tools/r42")
import numpy as np
import kfuse_model as km, kfuse_validate as kv
def mut(t, rnd):
    if rnd == "trunc":
        return np.floor(t)          # mutation: floor (toward -inf) instead of toward zero
    return np.rint(t)               # mutation: ties to even instead of away
km._quantise_term = mut
kv.v5(random.Random(3), 3)
print("mutated model flagged:", kv.FAILED)
