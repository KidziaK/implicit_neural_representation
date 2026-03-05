import numpy as np
from scipy.spatial import cKDTree

def chamfer_distance(points1: np.ndarray, points2: np.ndarray) -> float:
    tree1 = cKDTree(points1)
    tree2 = cKDTree(points2)
    
    dist1, _ = tree1.query(points2)
    dist2, _ = tree2.query(points1)
    
    return float(np.mean(dist1**2) + np.mean(dist2**2))

def hausdorff_distance(points1: np.ndarray, points2: np.ndarray) -> float:
    tree1 = cKDTree(points1)
    tree2 = cKDTree(points2)
    
    dist1, _ = tree1.query(points2)
    dist2, _ = tree2.query(points1)
    
    return float(max(np.max(dist1), np.max(dist2)))
