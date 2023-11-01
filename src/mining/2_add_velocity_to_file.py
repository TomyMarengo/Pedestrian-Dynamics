import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Read the merged txt file into a DataFrame
df = pd.read_csv('../../txt/merged_trajectories.txt', delim_whitespace=True, header=None, names=['Frame', 'Y', 'X', 'ID'])

# Time step between frames
dt = 4 / 30

# Create img/ folder if not exists
if not os.path.exists('img'):
    os.makedirs('img')

# Initialize a velocity column with NaN values
df['Velocity'] = np.nan

# Calculate velocity for each particle and save the plot
for uid in df['ID'].unique():
    ped_data = df[df['ID'] == uid].sort_values('Frame')
    
    dx = ped_data['X'].diff()  # Change in X
    dy = ped_data['Y'].diff()  # Change in Y
    
    # Calculate magnitude of velocity
    velocity = np.sqrt((dx / dt) ** 2 + (dy / dt) ** 2)
    
    # Update the velocity column in the main dataframe for the current particle
    df.loc[ped_data.index, 'Velocity'] = velocity.values
    
    plt.figure()
    plt.plot(ped_data['Frame'][1:], velocity[1:])
    plt.xlabel('Frame')
    plt.ylabel('Velocity Magnitude')
    plt.title(f'Particle {int(uid)} Velocity vs Time')
    plt.savefig(f'img/velocity_trajectory_{int(uid)}.png')
    plt.close()

# Save the y column multiplied by -1
df['Y'] *= -1

# Save the updated DataFrame with the velocity column back to a .txt file
df.to_csv('../../txt/merged_trajectories_with_velocity.txt', sep='\t', index=False, header=False)