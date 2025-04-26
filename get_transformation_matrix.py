def augmented_coord(p):
    # creates augmented homogeneous coordinates given the point x
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
 
"""Get transformation matrix defining the relative drone motion (rotation and translation) w.r.t previous frame'
from a  given Fundamental matrix F, and the intrinsic matrix K, using Singular Value Decomposition (SVD)."""
def get_transformation_matrix(F, K):
    # Compute essential matrix from fundamental matrix and intrinsics
    E = K.T @ F @ K
    
    # Define Matrix W to find F
    W = np.mat([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
     
    # SVD to find the Fundamental matrix F
    U, _, V = np.linalg.svd(E)
    assert np.linalg.det(U) > 0
 
    # Make sure U and V's det is positive for it to be a rotation matrix
    if np.linalg.det(U) < 0:
        U *= -1
    if np.linalg.det(V) < 0:
        V *= -1
 
    # Compute the rotation matrix
    R = U @ W @ Vt
    if np.trace(R) < 0:
        R = U @ W.T @ V
 
    # Get translation vector t from U (3rd column)
    t = U[:, 2]
 
    # Define the T transformation matrix
    T = np.eye(4)
 
    # T = [R|t]
    T[:3, :3] = R
    T[:3, 3] = t
 
    # Return the 4x4 homogeneous transformation matrix representing the pose
    return T
