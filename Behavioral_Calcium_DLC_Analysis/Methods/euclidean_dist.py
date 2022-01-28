import math
import numpy as np


def is_same_vector_dim(p, q) -> bool:
    if len(p) == len(q):
        return True
    else:
        return False


def vectors_dim(p, q):
    length = None
    if is_same_vector_dim(p, q) == True:
        length = len(p)
    else:
        pass

    return length


def squared_dist(p_ith, q_ith) -> float:
    return ((q_ith - p_ith)**2)


def euclid_dist(p, q) -> float:
    n = vectors_dim(p, q)
    sum = 0

    try:
        for i in range(n):
            sum += squared_dist(p[i], q[i])

        return math.sqrt(sum)
    except TypeError:
        print(f"Vectors are not of same dimensions!")


def euclid_dist_alex(t1, t2):
    return math.sqrt(sum((t1-t2)**2))


p = np.asarray([1, 2, 5])
q = np.asarray([2, 3, 1])

res = euclid_dist(p, q)
print(f"Euclidean distance = {res}")

###### Naive Approach ######
# Now apply this to our neural activity data
# 1) get the smoothed, z-scored csv.
# 2) For all combinations of pairing by 2, get those euclidean distance values
# 3) Determine intervals if distance values in which we can group signals

# ^But we we're going to cluster, might as well code up the kmeans while using these functions, not this naive approach
# This euclid_dist function works for 1d data (one single time point or a group of time points)
