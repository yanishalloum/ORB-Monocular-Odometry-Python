import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from test_matching import ORB_feature_detector, generate_orb_like_descriptors

# === Folder path ===
img_dir = "./Datasets/EuRoc/MH01/mav0/cam0/data"
start_ts = 1403636625913555456
end_ts = 1403636643563555584

# === Camera Intrinsics from EuRoC ===
K = np.array([
    [458.654, 0, 367.215],
    [0, 457.296, 248.375],
    [0, 0, 1]
])

# === Get and sort image filenames ===
all_files = sorted([
    f for f in os.listdir(img_dir)
    if f.endswith(".png") and start_ts <= int(f[:-4]) <= end_ts
], key=lambda x: int(x[:-4]))

# === Initialize pose: identity matrix ===
trajectory = [np.zeros((3, 1))]  # Initial position at origin
current_R = np.eye(3)
current_t = np.zeros((3, 1))

# === Process image pairs ===
for i in range(len(all_files) - 1):
    img1_path = os.path.join(img_dir, all_files[i])
    img2_path = os.path.join(img_dir, all_files[i + 1])

    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    kp1 = ORB_feature_detector(img1)
    kp2 = ORB_feature_detector(img2)

    desc1, kp1_valid = generate_orb_like_descriptors(img1, kp1)
    desc2, kp2_valid = generate_orb_like_descriptors(img2, kp2)

    kp1_cv = [cv2.KeyPoint(x, y, 1) for x, y in kp1_valid]
    kp2_cv = [cv2.KeyPoint(x, y, 1) for x, y in kp2_valid]

    if len(desc1) == 0 or len(desc2) == 0:
        print(f"[{all_files[i]} ➜ {all_files[i+1]}] Skipping: No descriptors.")
        continue

    desc1_cv = np.packbits(desc1, axis=1)
    desc2_cv = np.packbits(desc2, axis=1)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc1_cv, desc2_cv)
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) < 8:
        print(f"[{all_files[i]} ➜ {all_files[i+1]}] Skipping: Not enough matches.")
        continue

    pts1 = np.float32([kp1_cv[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2_cv[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    F, mask_F = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
    if F is None:
        print(f"[{all_files[i]} ➜ {all_files[i+1]}] Skipping: No fundamental matrix.")
        continue

    E = K.T @ F @ K
    retval, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)

    # === Update global pose ===
    scale = 1.0  # Assume scale = 1 (no scale info in monocular setup)
    current_t += scale * current_R @ t
    current_R = R @ current_R
    trajectory.append(current_t.copy())

# === Convert trajectory to numpy array for plotting ===
trajectory_np = np.hstack(trajectory)  # Shape: (3, N)

# === Plot trajectory ===
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(trajectory_np[0], trajectory_np[1], trajectory_np[2], label='Estimated Camera Trajectory', color='blue')
ax.scatter(trajectory_np[0, 0], trajectory_np[1, 0], trajectory_np[2, 0], color='green', label='Start')
ax.scatter(trajectory_np[0, -1], trajectory_np[1, -1], trajectory_np[2, -1], color='red', label='End')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title("Camera Trajectory from ORB-like Features")
ax.legend()
plt.show()
