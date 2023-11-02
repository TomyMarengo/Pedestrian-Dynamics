import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Read the merged txt file into a DataFrame
df = pd.read_csv('../../txt/merged_trajectories_with_velocity.txt', delim_whitespace=True, header=None, names=['Frame', 'Y', 'X', 'ID', 'Velocity'])

# Time step between frames
dt = 4 / 30

# Create img/ folder if not exists
if not os.path.exists('../../img'):
    os.makedirs('../../img')

# Initialize a velocity column with NaN values
df['Velocity'] = np.nan
df['vy'] = np.nan
df['vx'] = np.nan

# Calculate velocity for each particle and save the plot
for uid in df['ID'].unique():
    ped_data = df[df['ID'] == uid].sort_values('Frame')
    
    dx = ped_data['X'].diff()  # Change in X
    dy = ped_data['Y'].diff()  # Change in Y
    
    # Calculate magnitude of velocity
    velocity = np.sqrt((dx / dt) ** 2 + (dy / dt) ** 2)

    vx = dx/dt
    vy=-dy/dt
    
    # Update the velocity column in the main dataframe for the current particle
    df.loc[ped_data.index, 'vx'] = vx.values
    df.loc[ped_data.index, 'vy'] = vy.values
    df.loc[ped_data.index, 'Velocity'] = velocity.values

# Save the updated DataFrame with the velocity column back to a .txt file
df.to_csv('../../txt/merged_trajectories_with_vx_vy.txt', sep='\t', index=False, header=False)