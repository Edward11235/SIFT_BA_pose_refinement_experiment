import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

# Load images 
image_one_path = "/home/nerf-bridge/Desktop/OBA/data/chair.png" 
image_two_path = "/home/nerf-bridge/Desktop/OBA/data/chair_GT.png"  
img1 = cv2.imread(image_one_path, 0)  # Image 1 
img2 = cv2.imread(image_two_path, 0)  # Image 2

# Initialize ORB detector
orb = cv2.ORB_create()

# Find keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# Create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors
matches = bf.match(des1, des2)

# Calculate orientations and filter matches
orientations = []
for match in matches:
    pt1 = kp1[match.queryIdx].pt
    pt2 = kp2[match.trainIdx].pt

    # Calculate orientation (slope of line connecting keypoints)
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]

    orientation = math.atan2(dy, dx) if dx != 0 else np.pi / 2  # Avoid division by zero
    orientations.append(orientation)

# Sort orientations
orientations.sort()

# Define the threshold
threshold = np.deg2rad(1)  # Acceptable difference in radians, adjust as needed

# Find the interval that contains the most orientations
max_count = 0
common_orientation = 0
for i in range(len(orientations)):
    lower_bound = orientations[i]
    upper_bound = lower_bound + 2 * threshold
    count = sum(lower_bound <= orientation <= upper_bound for orientation in orientations)

    if count > max_count:
        max_count = count
        common_orientation = (lower_bound + upper_bound) / 2

# Filter matches with orientation within threshold of the most common orientation
matches_parallel = [match for match, orientation in zip(matches, orientations)
                    if abs(orientation - common_orientation) <= threshold]

# Draw matches with similar orientation
img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches_parallel, None, flags=2)

# Display the image
plt.imshow(img3), plt.show()

# Save the image
cv2.imwrite('matched_features_parallel.jpg', img3)

