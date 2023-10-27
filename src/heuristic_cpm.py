import math
import pandas as pd


# Read the merged txt file into a DataFrame
df = pd.read_csv('../txt/merged_trajectories_with_velocity.txt', delim_whitespace=True, header=None, names=['Frame', 'Y', 'X', 'ID', 'Velocity'])

#Get the time from frames
time = df['Frame'] * 4/30

# Get the velocity for each particle in for one frame
def particle_velocity(frame):
    if df['Frame'] == frame:
        particles = df.set_index('ID')['Velocity'].to_dict()
        return particles

              