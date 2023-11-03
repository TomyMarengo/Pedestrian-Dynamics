import cv2
import numpy as np

w = 9.75
h = 6.5
# Calibration points (coordinates in meters and coordinates in pixels)
world_points = np.array([
    [-w, -h], 
    [-w, h],
    [w, -h],
    [w, h],
], dtype=np.float32)

# Array of target points
image_points = np.array([
    # Bottom Left
    [226, 610],
    # Top Left
    [234, 150],
    # Bottom Right
    [920, 598],
    # Top Right
    [922, 158],
], dtype=np.float32)

# Calculate the perspective projection matrix
matrix = cv2.getPerspectiveTransform(world_points, image_points)
print(matrix.tolist())

# Transform the coordinates of a point in meters to coordinates in pixels
# point_in_meters = np.array([[]], dtype=np.float32)
# transformed_point = cv2.perspectiveTransform(point_in_meters, matrix)
