import cv2
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_distances

def cosine_similarity_matcher(des1, des2, ratio_threshold=0.75):
    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        return []

    distance_matrix = cosine_distances(des1, des2)  # Shape: (len(des1), len(des2))

    matches = []
    for i in range(distance_matrix.shape[0]):
        sorted_indices = np.argsort(distance_matrix[i])  # Sort indices based on distance
        best_idx = sorted_indices[0]
        second_best_idx = sorted_indices[1]

        best_dist = distance_matrix[i][best_idx]
        second_best_dist = distance_matrix[i][second_best_idx]

        # Lowe's ratio test
        if best_dist < ratio_threshold * second_best_dist:
            match = cv2.DMatch()
            match.queryIdx = i
            match.trainIdx = best_idx
            match.distance = best_dist
            matches.append(match)

    return matches


def stitch(img1, img2, output_path):
    """Stitch two images using SIFT features"""

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create(nfeatures=500)

    print("Detecting SIFT features...")
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    print(f"Found {len(kp1)} keypoints in first image and {len(kp2)} in second image")

    good_matches = cosine_similarity_matcher(des1, des2)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Find homography matrix
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Calculate size of the stitched image
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Calculate the dimensions of the stitched image
    corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)

    # Transform img1 corners to img2 space
    warped_corners1 = cv2.perspectiveTransform(corners1, H)

    # Combine all corners to find min/max coordinates
    all_corners = np.concatenate((warped_corners1, corners2), axis=0)
    x_min = np.min(all_corners[:, 0, 0])
    y_min = np.min(all_corners[:, 0, 1])
    x_max = np.max(all_corners[:, 0, 0])
    y_max = np.max(all_corners[:, 0, 1])

    # Translation matrix to shift the image to positive coordinates
    translation = np.array([
        [1, 0, -x_min],
        [0, 1, -y_min],
        [0, 0, 1]
    ])

    # Combine homography with translation
    H_combined = translation @ H

    # Create the stitched image
    width = int(x_max - x_min)
    height = int(y_max - y_min)
    stitched = np.zeros((height, width, 3), dtype=np.uint8)

    # Warp first image
    warped1 = cv2.warpPerspective(img1, H_combined, (width, height))

    # Create a simple mask for the second image
    mask2 = np.zeros((height, width), dtype=np.uint8)
    cv2.warpPerspective(np.ones(img2.shape[:2], dtype=np.uint8), translation,
                        (width, height), dst=mask2)

    # Simple blending - place second image in stitched canvas
    for y in range(height):
        for x in range(width):
            if mask2[y, x] > 0:
                # Map to original img2 coordinates
                orig_x = x + int(x_min)
                orig_y = y + int(y_min)

                # Check if within img2 bounds
                if 0 <= orig_x < w2 and 0 <= orig_y < h2:
                    stitched[y, x] = img2[orig_y, orig_x]

    # Blend first warped image where second image is black
    mask = (stitched == 0)
    stitched[mask] = warped1[mask]

    print(f"Saving stitched image to {output_path}")
    cv2.imwrite(output_path, result)
    return result

    return stitched