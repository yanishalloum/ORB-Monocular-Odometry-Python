def augmented_coord(p):
    # creates augmented homogeneous coordinates given the point x
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
 
"""Get transformation matrix defining the relative drone motion (rotation and translation) w.r.t previous frame'
from a  given Fundamental matrix $F$ using Singular Value Decomposition (SVD)."""
def get_transformation_matrix(F):
    # Define Matrix W to find F
    W = np.mat([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
     
    # SVD to find the Fundamental matrix F
    U, d, V = np.linalg.svd(F)
    assert np.linalg.det(U) > 0
 
    # Make sure V's det is positive for it to be a rotation matrix
    if np.linalg.det(V) < 0:
        Vt *= -1
 
    # R = (U.W).V
    R = np.dot(np.dot(U, W), V)
 
    # If R's diagonal sum is positive, then it's a proper rotation matrix
    # If not, R is defined as (U.(W.T).V
    if np.sum(R.diagonal()) < 0:
        R = np.dot(np.dot(U, W.T), V)
 
    # Get translation vector t from U (3rd column)
    t = U[:, 2]
 
    # Define the T transformation matrix
    T_matrix = np.eye(4)
 
    # T = [R|t]
    T_matrix[:3, :3] = R
    T_matrix[:3, 3] = t
 
    print(d)
 
    # Return the 4x4 homogeneous transformation matrix representing the pose
    return T_matrix
