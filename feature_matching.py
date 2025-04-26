import cv2
import numpy as np
from feature_extract import ORB_feature_detector, generate_orb_like_descriptors  # Import the functions

def compute_orientation(image, x, y, patch_size=31):
    half = patch_size // 2
    patch = image[y - half:y + half + 1, x - half:x + half + 1]

    if patch.shape != (patch_size, patch_size):
        return 0

    m = cv2.moments(patch)
    if m['m00'] == 0:
        return 0
    cx = m['m10'] / m['m00']
    cy = m['m01'] / m['m00']
    angle = np.arctan2(cy - half, cx - half)
    return angle

# Images to compare
img1 = cv2.imread(r"C:\Users\User\Documents\ELE6209\test_orb\Datasets\EuRoc\MH01\mav0\cam0\data\1403636631463555584.png", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(r"C:\Users\User\Documents\ELE6209\test_orb\Datasets\EuRoc\MH01\mav0\cam0\data\1403636634563555584.png", cv2.IMREAD_GRAYSCALE)

# Detection and description 
kp1 = ORB_feature_detector(img1, N=12, T=20)
kp2 = ORB_feature_detector(img2, N=12, T=20)

desc1, kp1_valid = generate_orb_like_descriptors(img1, kp1)
desc2, kp2_valid = generate_orb_like_descriptors(img2, kp2)

# Convert to KeyPoint objects
kp1_cv = [cv2.KeyPoint(x, y, 1) for x, y in kp1_valid]
kp2_cv = [cv2.KeyPoint(x, y, 1) for x, y in kp2_valid]

# Convert descriptors to OpenCV format
desc1_cv = np.packbits(desc1, axis=1)  # convert list of bits → bytes
desc2_cv = np.packbits(desc2, axis=1)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(desc1_cv, desc2_cv)
matches = sorted(matches, key=lambda x: x.distance)

img_match = cv2.drawMatches(img1, kp1_cv, img2, kp2_cv, matches[:50], None, flags=2)

cv2.imshow("ORB-like Matching", img_match)
cv2.waitKey(0)
cv2.destroyAllWindows()
