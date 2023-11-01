import cv2
import pandas as pd
import numpy as np

# Read the stable video 
cap = cv2.VideoCapture("C:\\Users\\desir\\IdeaProjects\\PedestrianDynamics\\gif\\stable_video.avi")

video_output = "C:\\Users\\desir\\IdeaProjects\\PedestrianDynamics\\gif\\video_output.avi"

# Create videoWriter
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec para AVI, ajusta según el formato deseado
fps = 30  # Velocidad de cuadros por segundo (ajusta según tus necesidades)
frame_size = (1100, 620) 
video_writer = cv2.VideoWriter(video_output, fourcc, fps, frame_size)

# Read the filtered_merged_file into a DataFrame
df = pd.read_csv('C:\\Users\\desir\\IdeaProjects\\PedestrianDynamics\\txt\\merged_trajectories_with_velocity.txt', delim_whitespace=True, header=None, names=['Frame', 'Y', 'X', 'ID', 'Velocity'])
df = df.reset_index()

# Convert coordinates to pixels from transformation_matrix.py 
transformation_matrix = np.array([
    [36.745838342956766, 0.7496593456071169, 583.1818099954976],
    [0.7626752955884007, -34.354527957409545, 378.0371004052229],
    [0.002277296469101962, 0.0006432109088570141, 1.0]
    ], dtype=np.float32)

frame_i = 1
inclination_angle = np.radians(5)

# Define a list of unique colors
unique_colors = [
    '#c0c0c0', '#2f4f4f', '#808000', '#483d8b', '#b22222', 
    '#9acd32', '#8b008b', '#48d1cc', '#ff0000', '#ff8c00',
    '#ffff00', '#00ff00', '#8a2be2', '#00ff7f', '#3cb371',
    '#00bfff', '#0000ff', '#ff00ff', '#1e90ff', '#000080',
    '#db7093', '#f0e68c', '#ff1493', '#ffa07a', '#ee82ee',
]

# Create a list of unique IDs and assign a unique color to each ID
unique_ids = df['ID'].unique()
colors = unique_colors

trail = 5  

# Dictionary to keep track of previous positions of particles
trail_lines = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    particles = df[df['Frame'] == frame_i]
    h, w = frame.shape[:2]

    for uid, color in zip(unique_ids, colors):

        # Convert color to BGR
        color_bgr = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))

        particle = particles[particles['ID'] == uid]

        x = particle['X'].values[0] * np.cos(inclination_angle)
        y = particle['Y'].values[0] * np.cos(-inclination_angle)
        x, y, _ = transformation_matrix @ np.array([x, y, 1])

        if uid not in trail_lines:
            trail_lines[uid] = []

        trail_lines[uid].append((int(x), int(y)))

        if len(trail_lines[uid]) > trail:
            trail_lines[uid].pop(0)

        # Draw trail line
        for i in range(len(trail_lines[uid]) - 1):
            pt1 = trail_lines[uid][i]
            pt2 = trail_lines[uid][i + 1]
            frame = cv2.line(frame, pt1, pt2, color_bgr, 2)

        frame = cv2.circle(frame, (int(x), int(y)), 10, color_bgr, -1)

    frame_resized = cv2.resize(frame, (1100, 620))
    cv2.imshow("Frame", frame_resized)

    video_writer.write(frame_resized)

    # Advance with space cv2.waitKey(0), close with q
    if cv2.waitKey(1) == ord('q'):
        break

    frame_i += 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_i * 4)

cap.release()

# Save video
video_writer.release()
cv2.destroyAllWindows()