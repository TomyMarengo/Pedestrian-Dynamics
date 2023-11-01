import cv2
import pandas as pd
import math
import numpy as np


#cap = cv2.VideoCapture("../Targets_720_0_1000.mov")
cap = cv2.VideoCapture("../stable_video.avi")
df = pd.read_csv('../txt/merged_trajectories_with_velocity.txt', delim_whitespace=True, header=None, names=['Frame', 'Y', 'X', 'ID', 'Velocity'])
df = df.reset_index()

#para obtener la matriz --> transformation_matrix.py , poner las coordenadas en pixeles de los puntos
transformation_matrix = np.array([
    [36.745838342956766, 0.7496593456071169, 583.1818099954976],
    [0.7626752955884007, -34.354527957409545, 378.0371004052229],
    [0.002277296469101962, 0.0006432109088570141, 1.0]
    ], dtype=np.float32)


frame_i = 1

while True:
    ret, frame = cap.read()
    if not ret:
        break

    particles = df[df['Frame'] == frame_i]
    h, w = frame.shape[:2]

    for index, particle in particles.iterrows():
        x, y = particle['X'], particle['Y']
        x, y, _ = transformation_matrix @ np.array([x, y, 1])

        id = particle['ID']

        frame = cv2.circle(frame, (int(x), int(y)), 10, (255, 0, 0), -1)

    cv2.imshow("Frame", frame)
    #para que se quede parado, avanzar con tecla (espacio): cv2.waitKey(0)
    if cv2.waitKey(1) == ord('q'):
        break

    frame_i += 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_i * 4)

cap.release()
cv2.destroyAllWindows()