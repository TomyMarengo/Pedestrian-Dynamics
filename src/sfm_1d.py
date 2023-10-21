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
to_frame = 65
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
    plt.plot(ped_data['Frame']*4/30, ped_data['Velocity'])

def sfm_1D(tau, vd, from_frame, to_frame, m = 1):
    """Simulates the Social Force Model in 1D for a single particle."""
    # Initialization
    num_steps = (to_frame - from_frame + 1 ) * 100
    v = np.zeros(num_steps)

    # Simulation Loop
    for i in range(1, num_steps):
        # Compute the autopropulsion force
        f = m * (vd - v[i-1]) / tau
        # Compute the acceleration
        a = f / m
        # Update the velocity using the acceleration
        v[i] = v[i-1] + a * dt/100

    return v

# GRAPH PLOTS
velocity_trajectory(uid)

# Plotting SFM with different tau values
t = np.arange(from_frame, to_frame + 1, 1)
t = t * 4 / 30

for tau, color in zip(taus, colors):
    v = sfm_1D(tau, vd, from_frame, to_frame)
    # Take velocities one every 100 steps
    v = v[::100]
    # Only dots, no lines
    plt.plot(t, v, label=f'SFM tau={tau}', color=color, marker='o', markersize=4 , linestyle='None')

plt.xlabel('Tiempo (s)')
plt.ylabel('Velocidad $(\\frac{m}{s})$')
plt.title(f'Agente {int(uid)}')  # TODO: Quitar esto
plt.legend()
plt.savefig(f'../img/velocity_tau_{int(uid)}.png')
plt.show()
print(f'Graph saved at ../img/velocity_tau_{int(uid)}.png')
