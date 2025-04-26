import cv2
import numpy as np

def ORB_feature_detector(image, N=12, T=20):
    h, w = image.shape
    keypoints = []

    circle_offsets = [
        (0, -3), (1, -3), (2, -2), (3, -1),
        (3, 0), (3, 1), (2, 2), (1, 3),
        (0, 3), (-1, 3), (-2, 2), (-3, 1),
        (-3, 0), (-3, -1), (-2, -2), (-1, -3)
    ]

    for y in range(3, h-3):
        for x in range(3, w-3):
            I_p = int(image[y, x])
            circle_values = [int(image[y + dy, x + dx]) for dx, dy in circle_offsets]

            brighter = [val > I_p + T for val in circle_values]
            darker   = [val < I_p - T for val in circle_values]

            def has_n_consecutive(arr, N):
                extended = arr + arr[:N-1]
                count = 0
                for val in extended:
                    if val:
                        count += 1
                        if count >= N:
                            return True
                    else:
                        count = 0
                return False

            if has_n_consecutive(brighter, N) or has_n_consecutive(darker, N):
                keypoints.append((x, y))

    return keypoints

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

def generate_orb_like_descriptors(image, keypoints, patch_size=31, n_bits=256):
    half = patch_size // 2
    descs = []
    valid_kp = []

    np.random.seed(42)
    pairs = np.random.randint(-half, half + 1, size=(n_bits, 4))  # (x1, y1, x2, y2)

    for x, y in keypoints:
        if x - half < 0 or y - half < 0 or x + half >= image.shape[1] or y + half >= image.shape[0]:
            continue

        angle = compute_orientation(image, x, y, patch_size)
        cos_a, sin_a = np.cos(angle), np.sin(angle)

        patch = image[y - half:y + half + 1, x - half:x + half + 1]
        desc = []

        for dx1, dy1, dx2, dy2 in pairs:
            # Rotate pairs
            rx1 = int(cos_a * dx1 - sin_a * dy1)
            ry1 = int(sin_a * dx1 + cos_a * dy1)
            rx2 = int(cos_a * dx2 - sin_a * dy2)
            ry2 = int(sin_a * dx2 + cos_a * dy2)

            px1 = half + rx1
            py1 = half + ry1
            px2 = half + rx2
            py2 = half + ry2

            if 0 <= px1 < patch_size and 0 <= py1 < patch_size and 0 <= px2 < patch_size and 0 <= py2 < patch_size:
                val1 = patch[py1, px1]
                val2 = patch[py2, px2]
                desc.append(1 if val1 < val2 else 0)

        if len(desc) == n_bits:
            descs.append(desc)
            valid_kp.append((x, y))

    return np.array(descs, dtype=np.uint8), valid_kp


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
