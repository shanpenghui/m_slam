import numpy as np
from scipy.spatial.transform import Rotation

def rigid_transform_3D(A, B):
    
    assert len(A) == len(B)

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    AA = A - centroid_A
    BB = B - centroid_B

    H = np.dot(np.transpose(AA), BB)

    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    if np.linalg.det(R) < 0:
        Vt[2,:] *= -1
        R = np.dot(Vt.T, U.T)

    t = np.dot(-R, centroid_A) + centroid_B

    return R, t