import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# CHANGE UID HERE
uid = 9
# CHANGE PARAMETERS HERE
taus = [0.5, 1.0, 1.5, 2.0]  # Example tau values to try out
colors = ['red', 'green', 'purple', 'orange']
from_frame = 51
to_frame = 75
vd = 1.4
# Time step between frames
dt = 4 / 30
# Read the merged txt file into a DataFrame
df = pd.read_csv('../txt/merged_trajectories_with_velocity.txt', delim_whitespace=True, header=None, names=['Frame', 'Y', 'X', 'ID', 'Velocity'])

# Create img/ folder if not exists
if not os.path.exists('../img'):
    os.makedirs('../img')

# Calculate velocity for one particle
def velocity_trajectory(uid):
    ped_data = df[df['ID'] == uid].sort_values('Frame')
    plt.figure()
    plt.plot(ped_data['Frame'], ped_data['Velocity'])
    plt.xlabel('Frame')
    plt.ylabel('Velocity Magnitude')
    plt.title(f'Particle {int(uid)} Velocity vs Time')

def sfm_1D(tau, vd, from_frame, to_frame, m = 1):
    """Simulates the Social Force Model in 1D for a single particle."""
    # Initialization
    num_steps = to_frame - from_frame + 1
    v = np.zeros(num_steps)

    # Simulation Loop
    for i in range(1, num_steps):
        # Compute the autopropulsion force
        f = m * (vd - v[i-1]) / tau
        # Compute the acceleration
        a = f / m
        # Update the velocity using the acceleration
        v[i] = v[i-1] + a * dt

    return v

# GRAPH PLOTS
velocity_trajectory(uid)

# Plotting SFM with different tau values
t = np.arange(from_frame, to_frame + 1, 1)
for tau, color in zip(taus, colors):
    v = sfm_1D(tau, vd, from_frame, to_frame)
    plt.plot(t, v, label=f'SFM tau={tau}', color=color)

plt.title('Particle Velocity vs Time')
plt.xlabel('Time (frames)')
plt.ylabel('Velocity Magnitude')
plt.legend()
plt.savefig(f'../img/velocity_tau_{int(uid)}.png')
print(f'Graph saved at ../img/velocity_tau_{int(uid)}.png')
