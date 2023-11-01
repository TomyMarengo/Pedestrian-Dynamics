import pandas as pd
import matplotlib.pyplot as plt
import os

# Read the txt file with velocities into a DataFrame
df = pd.read_csv('../../txt/merged_trajectories_with_velocity.txt', delim_whitespace=True, header=None, names=['Frame', 'Y', 'X', 'ID', 'Velocity'])

# Create img/ folder if not exists
if not os.path.exists('img'):
    os.makedirs('img')

# Plot velocity vs frame for each particle and save the plot
for uid in df['ID'].unique():
    ped_data = df[df['ID'] == uid].sort_values('Frame')
    
    plt.figure()
    plt.plot(ped_data['Frame']*4/30, ped_data['Velocity'])
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Velocidad $(\\frac{m}{s})$')
    plt.title(f'Agente {int(uid)}')  # TODO: Quitar esto
    plt.savefig(f'img/velocity_trajectory_{int(uid)}.png')
    plt.close()
    
print('Done!')