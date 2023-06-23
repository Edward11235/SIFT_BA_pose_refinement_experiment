import numpy as np
from scipy.optimize import least_squares
import cv2


np.random.seed(42)
# Define function to transform 3D points to 2D
def project(points, params):
    x0, y0, f = params[:3]
    rotation_vector = params[3:6].astype(np.float64)  # Ensure rotation vector is float64
    translation_vector = params[6:]

    R, _ = cv2.Rodrigues(rotation_vector)
    projected_points = np.dot(points, R.T) + translation_vector
    projected_points = projected_points[:, :2] / projected_points[:, 2, np.newaxis]
    projected_points *= f
    projected_points[:, 0] += x0
    projected_points[:, 1] += y0
    return projected_points

# Define reprojection error
def fun(params, n, m, points_2d, points_3d):
    projected_points = project(points_3d, params)
    return (projected_points - points_2d).ravel()

# Assume you have some 3D points
points_3d = np.random.rand(50, 3)

# And you know the true parameters
true_params = np.array([320, 240, 1000, 0, 0, 0, 0, 0, 0], dtype=np.float64)  # Ensure params are float64

# And you have 2D points by projecting 3D points using the true parameters
points_2d = project(points_3d, true_params)
points_2d_with_noise = points_2d.copy()
points_2d_with_big_noise = points_2d.copy()
points_2d_with_five_nonsense_point = points_2d.copy()

for point in points_2d_with_noise:
    point[0] += np.random.normal(0,1)
    point[1] += np.random.normal(0,1)
    
for point in points_2d_with_big_noise:
    point[0] += np.random.normal(0,10)
    point[1] += np.random.normal(0,10)

for point in points_2d_with_five_nonsense_point[-5:]:
    point[0] += np.random.normal(0,1000)
    point[1] += np.random.normal(0,1000)

# You want to find these parameters by bundle adjustment
initial_estimation = np.array([300, 200, 900, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float64)  # Ensure params are float64

# Use the least_squares function to solve the bundle adjustment problem
res = least_squares(fun, initial_estimation, args=(2, 3, points_2d, points_3d))
res_with_noise = least_squares(fun, initial_estimation, args=(2, 3, points_2d_with_noise, points_3d))
res_with_big_noise = least_squares(fun, initial_estimation, args=(2, 3, points_2d_with_big_noise, points_3d))
res_with_five_nonsense = least_squares(fun, initial_estimation, args=(2, 3, points_2d_with_five_nonsense_point, points_3d))

# Print the true parameters and the ones estimated by bundle adjustment
print('True pose: ', true_params)
print('#######################################')
print('initial pose: ', initial_estimation)
print('#######################################')
print('Estimated pose: ', res.x)
print('#######################################')
print('Estimated pose when noise occurs: ', res_with_noise.x)
print('#######################################')
print('Estimated pose when big noise occurs: ', res_with_big_noise.x)
print('#######################################')
print('Estimated pose with five nonsense point: ', res_with_five_nonsense.x)