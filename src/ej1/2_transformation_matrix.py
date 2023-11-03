import cv2
import numpy as np

w = 9.75
h = 6.5
# Puntos de calibración (coordenadas en metros y coordenadas en píxeles)
world_points = np.array([
    [-w, -h], 
    [-w, h],
    [w, -h],
    [w, h],
], dtype=np.float32)
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

# Calcular la matriz de proyección perspectiva
matrix = cv2.getPerspectiveTransform(world_points, image_points)
print(matrix.tolist())

# Transformar las coordenadas de un punto en metros a coordenadas en píxeles
# point_in_meters = np.array([[]], dtype=np.float32)
# transformed_point = cv2.perspectiveTransform(point_in_meters, matrix)

# transformed_point ahora contiene las coordenadas en píxeles del punto en la imagen
