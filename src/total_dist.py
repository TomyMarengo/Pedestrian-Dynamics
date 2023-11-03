import math
import pandas as pd
import numpy as np


df = pd.read_csv('../txt/merged_trajectories_with_velocity.txt', delim_whitespace=True, header=None, names=['Frame', 'Y', 'X', 'ID', 'Velocity'])


def dist(x1, y1, x2, y2 ):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

distancias = {}

for uid in df['ID'].unique():
    ped_data = df[df['ID'] == uid].sort_values('Frame')
    total_distance = 0.0 
    
    prev_x, prev_y = None, None
    for index, frame in ped_data.iterrows():
        x, y = frame['X'], frame['Y']
        if prev_x is not None and prev_y is not None:
            total_distance += dist(prev_x, prev_y, x, y)
        
        prev_x, prev_y = x, y
    
    uid_str = str(uid)
    distancias[uid_str] = total_distance

# Imprime la tabla
print("ID\tDistancia Total")
print("-" * 20)

for uid, distancia in distancias.items():
    print(f"{uid}\t{distancia:.2f}")

    