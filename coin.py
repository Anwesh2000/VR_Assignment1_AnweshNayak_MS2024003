import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

coin_path_dict = {
    "3": "coin_input/3_coin.jpeg",
    "13": "coin_input/13_coin.jpeg",
    "32": "coin_input/32_coin.jpeg",
    "73": "coin_input/73_coin.jpeg",
}

def detect_and_count_coins(image_path, min_dist, minR, maxR, blur=True):
    img = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if blur:
        blur_img = cv2.GaussianBlur(img_gray, (9, 9), 2)
    else:
        blur_img = img_gray

    circles = cv2.HoughCircles(
        blur_img,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=min_dist,
        param1=50,
        param2=30,
        minRadius=minR,
        maxRadius=maxR
    )

    # Create visualizations
    edge_vis = img.copy()
    segmentation_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    individual_coins = []

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Draw the outer circle on detection visualization
            cv2.circle(edge_vis, (i[0], i[1]), i[2], (0, 255, 0), 2)

            # Draw filled circle on mask (single channel)
            cv2.circle(segmentation_mask, (i[0], i[1]), i[2], 255, -1)

            # Extract individual coin
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.circle(mask, (i[0], i[1]), i[2], 255, -1)
            coin = cv2.bitwise_and(img, img, mask=mask)

            # Get bounding rectangle
            x = max(0, i[0] - i[2])
            y = max(0, i[1] - i[2])
            w = min(img.shape[1] - x, 2 * i[2])
            h = min(img.shape[0] - y, 2 * i[2])

            coin_cropped = coin[int(y):int(y + h), int(x):int(x + w)]
            if coin_cropped.size > 0:
                individual_coins.append(coin_cropped)

    # Create segmented visualization (coins in color, background black)
    segmented_vis = img.copy()
    segmented_vis[segmentation_mask == 0] = [0, 0, 0]

    coin_count = len(individual_coins)
    return img, edge_vis, segmented_vis, blur_img, coin_count

def visualize_results(original, blur, edges, segmented, count, expected_count, save_path=None):
    """
    Visualize the results with improved layout.
    Displays:
      - Original image with detection circles
      - Blurred image
      - Image with detected coins (edge visualization)
      - Segmented coins image
    If save_path is provided, the figure is saved to that file.
    """
    plt.figure(figsize=(10, 10))

    # Original image
    plt.subplot(321)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title(f'Original Image, Number of coins: {expected_count}')
    plt.axis('off')

    # Blur image (if grayscale, you may use cmap='gray')
    plt.subplot(322)
    plt.imshow(cv2.cvtColor(blur, cv2.COLOR_BGR2RGB))
    plt.title('Gaussian Blur for noise reduction')
    plt.axis('off')

    # Edge detection result
    plt.subplot(323)
    plt.imshow(cv2.cvtColor(edges, cv2.COLOR_BGR2RGB))
    plt.title(f'Number of coins detected: {count}')
    plt.axis('off')

    # Segmentation result
    plt.subplot(324)
    plt.imshow(cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB))
    plt.title('Segmented Coins')
    plt.axis('off')

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    # Parse command line arguments for the expected number of coins
    parser = argparse.ArgumentParser(description="Coin detection using HoughCircles.")
    parser.add_argument("--num_coins", type=int, default=32,
                        help="Expected number of coins in the image. Default is 32.")
    args = parser.parse_args()

    num_coins = args.num_coins

    # Set parameters based on expected number of coins
    if num_coins == 73:
        min_dist = 30
        minR = 15
        maxR = 60
    else:
        min_dist = 60
        minR = 25
        maxR = 95

    blur = True
    # Look up the image based on num_coins; if not found, default to "32"
    image_path = coin_path_dict.get(str(num_coins), coin_path_dict["32"])

    original, edges, segmented, blur_img, count = detect_and_count_coins(
        image_path, min_dist, minR, maxR, blur
    )
    save_path = f"coin_results/{str(num_coins)}_coins_output.png"
    print(f"\nTotal number of coins in the image: {num_coins}")
    print(f"Total number of coins detected:     {count}")
    visualize_results(original, blur_img, edges, segmented, count, num_coins, save_path=save_path)
