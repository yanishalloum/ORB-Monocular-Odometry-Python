import cv2
import numpy as np
import matplotlib.pyplot as plt

def ORB_feature_detector(image, N=12, T=20):
    h, w = image.shape
    keypoints = []

    # Coordonnées des 16 pixels sur le cercle de rayon 3 autour de p
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

            # Définir seuil haut et bas
            brighter = [val > I_p + T for val in circle_values]
            darker   = [val < I_p - T for val in circle_values]

            # Fonction utilitaire pour vérifier N consécutifs (avec wrap-around)
            def has_n_consecutive(arr, N):
                extended = arr + arr[:N-1]  # wrap around
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

if __name__ == "__main__":
    # Test avec une image
    image_path = "./Datasets/EuRoc/MH01/mav0/cam0/data/1403636579763555584.png"
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Détection manuelle
    features = ORB_feature_detector(img, N=11, T=30)

    # Affichage
    img_vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for x, y in features: 
        cv2.circle(img_vis, (x, y), 2, (0, 255, 0), -1)

    cv2.imshow("ORB Features", img_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
