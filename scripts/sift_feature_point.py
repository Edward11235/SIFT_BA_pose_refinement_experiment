import cv2
from matplotlib import pyplot as plt

# Load images 
image_one_path = "/home/nerf-bridge/Desktop/OBA/experiment_images/image_render" 
image_two_path = "/home/nerf-bridge/Desktop/OBA/experiment_images/image_GT"  
img1 = cv2.imread(image_one_path, 0)  # Image 1 
img2 = cv2.imread(image_two_path, 0)  # Image 2

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Find keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# Create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Match descriptors
matches = bf.match(des1, des2)

# Sort them in ascending order of distance
matches = sorted(matches, key=lambda x: x.distance)

# Draw first 50 matches
img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:30], None, flags=2)

# Display the image
plt.imshow(img3), plt.show()

# Save the image
cv2.imwrite('matched_features.jpg', img3)
