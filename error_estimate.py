
import numpy as np
from scipy.spatial import cKDTree

def calc_trans_error(ground_truth, estimated_traj):
    tree = cKDTree(estimated_traj)
    distance, index = tree.query(ground_truth)

    return distance