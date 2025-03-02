import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
import matplotlib.pyplot as plt

def cosine_similarity_matcher(des1, des2, ratio_threshold=0.75):
    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        return []

    distance_matrix = cosine_distances(des1, des2)  # Shape: (len(des1), len(des2))

    matches = []
    for i in range(distance_matrix.shape[0]):
        sorted_indices = np.argsort(distance_matrix[i])  # sort indices based on distance
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

def draw_matches(img1, img2, kp1, kp2, matches, output_path=None):
    # Create a new image showing matches between the images
    matched_img = cv2.drawMatches(
        img1, kp1,
        img2, kp2,
        matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        matchColor=(0, 255, 0)  # Green color for matches
    )

    # Add title text to the matched image
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(matched_img, 'Matches', (matched_img.shape[1] // 2 - 50, 30),
                font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Save visualization if output_path is provided
    if output_path:
        cv2.imwrite(output_path, matched_img)
        print(f"Match visualization saved to {output_path}")

    return matched_img

def stitch_with_visualization(img1_path, img2_path, output_path, match_viz_path=None):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create(nfeatures=500)

    print("Detecting SIFT features...")
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    print(f"Found {len(kp1)} keypoints in first image and {len(kp2)} in second image")

    good_matches = cosine_similarity_matcher(des1, des2)
    print(f"Found {len(good_matches)} good matches between images")

    # Create match visualization if requested
    if match_viz_path:
        match_img = draw_matches(img1, img2, kp1, kp2, good_matches, match_viz_path)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Find homography matrix using RANSAC
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Calculate the size of the stitched image
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)

    warped_corners1 = cv2.perspectiveTransform(corners1, H)
    all_corners = np.concatenate((warped_corners1, corners2), axis=0)
    x_min = np.min(all_corners[:, 0, 0])
    y_min = np.min(all_corners[:, 0, 1])
    x_max = np.max(all_corners[:, 0, 0])
    y_max = np.max(all_corners[:, 0, 1])

    translation = np.array([
        [1, 0, -x_min],
        [0, 1, -y_min],
        [0, 0, 1]
    ])
    H_combined = translation @ H

    width = int(x_max - x_min)
    height = int(y_max - y_min)
    stitched = np.zeros((height, width, 3), dtype=np.uint8)

    # Warp first image using the combined homography
    warped1 = cv2.warpPerspective(img1, H_combined, (width, height))

    # Create a mask for the second image
    mask2 = np.zeros((height, width), dtype=np.uint8)
    cv2.warpPerspective(np.ones(img2.shape[:2], dtype=np.uint8), translation,
                        (width, height), dst=mask2)

    # Simple blending: fill in pixels from img2 where mask2 is positive
    for y in range(height):
        for x in range(width):
            if mask2[y, x] > 0:
                orig_x = x + int(x_min)
                orig_y = y + int(y_min)
                if 0 <= orig_x < w2 and 0 <= orig_y < h2:
                    stitched[y, x] = img2[orig_y, orig_x]

    # Use the warped first image where the stitched image is still black
    mask_final = (stitched == 0)
    stitched[mask_final] = warped1[mask_final]

    print(f"Saving stitched image to {output_path}")
    cv2.imwrite(output_path, stitched)

    return stitched

# Input file paths
img1_path = "panorama_input/1.jpeg"
img2_path = "panorama_input/2.jpeg"
output_path = "panorama_results/stitched.jpg"
match_viz_path = "panorama_results/matches.jpg"

# Stitch images and create match visualization
stitched = stitch_with_visualization(img1_path, img2_path, output_path, match_viz_path)

# Display output images using matplotlib
# Read the saved images and convert them from BGR to RGB for display
stitched_disp = cv2.cvtColor(cv2.imread(output_path), cv2.COLOR_BGR2RGB)
match_disp = cv2.cvtColor(cv2.imread(match_viz_path), cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(match_disp)
plt.title("Matches Visualization")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(stitched_disp)
plt.title("Stitched Image")
plt.axis("off")

plt.tight_layout()
plt.show()
