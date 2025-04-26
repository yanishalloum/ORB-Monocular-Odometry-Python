import numpy as np

def augment_coord(p):
    """Add a column of ones to a 2D projected point (augmented coordinates)."""
    return np.hstack([p, np.ones((pts.shape[0], 1))])
 
def triangulate(pose1, pose2, pts1, pts2):
    # Initialize the result array to store the augmented coordinates of the 3D points
    3D_res = np.zeros((pts1.shape[0], 4))
 
    # define the projection matrices
    pose1 = np.linalg.inv(pose1)
    pose2 = np.linalg.inv(pose2)
 
    for i, p in enumerate(zip(augment_coord(pts1), augment_coord(pts2))):
        A = np.zeros((4, 4))
 
        A[0] = p[0][0] * pose1[2] - pose1[0]
        A[1] = p[0][1] * pose1[2] - pose1[1]
        A[2] = p[1][0] * pose2[2] - pose2[0]
        A[3] = p[1][1] * pose2[2] - pose2[1]
 
        # SVD on A
        _, _, vt = np.linalg.svd(A)
 
        # Solution the smallest singular value
        3D_res[i] = vt[3]
 
    # Return the 3D points in homogeneous coordinates
    return 3D_res
