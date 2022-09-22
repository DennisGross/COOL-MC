from common.adversarial_attacks.adversarial_attack import AdversarialAttack
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch
import math
from numpy import linalg as LA
import itertools

def cartesian_coord(*arrays, epsilon, feature_map=None):
    grid = np.meshgrid(*arrays)      
    coord_list = [entry.ravel() for entry in grid]
    points = np.vstack(coord_list).T
    if feature_map!=None:
        for col in range(points.shape[1]):
            target_feature = col in feature_map.values()
            if target_feature==False:
                points=points[~(points[:,col] != 0),:]

    return points[np.absolute(points).sum(axis=1) <= epsilon,:]

state = np.zeros(10, dtype=np.int8)
epsilon = 2
arr = cartesian_coord(*len(state)*[np.arange(-epsilon, epsilon+1)], epsilon=epsilon)
print(arr)