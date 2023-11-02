import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import numpy as np
import os

# Create img/ folder if not exists
if not os.path.exists('gif'):
    os.makedirs('gif')
if not os.path.exists('img'):
    os.makedirs('img')
for i in range(26):
    if not os.path.exists('img/' + str(i)):
        os.makedirs('img/' + str(i))

# Define a list of unique colors
unique_colors = [
    '#c0c0c0', '#2f4f4f', '#808000', '#483d8b', '#b22222', 
    '#9acd32', '#8b008b', '#48d1cc', '#ff0000', '#ff8c00',
    '#ffff00', '#00ff00', '#8a2be2', '#00ff7f', '#3cb371',
    '#00bfff', '#0000ff', '#ff00ff', '#1e90ff', '#000080',
    '#db7093', '#f0e68c', '#ff1493', '#ffa07a', '#ee82ee',
]

# Read the filtered_merged_file into a DataFrame
df = pd.read_csv('../../txt/merged_trajectories_with_velocity.txt', delim_whitespace=True, header=None, names=['Frame', 'Y', 'X', 'ID', 'Velocity'])
df_virtual = pd.read_csv('../../txt/virtual_pedestrian_trajectory.txt', delim_whitespace=True, header=None, names=['Frame', 'Y', 'X', 'VelX', 'VelY', 'TargetY', 'TargetX'])

# Create a list of unique IDs and assign a unique color to each ID
unique_ids = df['ID'].unique()
colors = unique_colors

# Create a dictionary to keep track of trails
trail_dict = {}

# Initialize the plot
fig, ax = plt.subplots()
scatter_dots = []

trail_lines = []  # Store trail lines to remove them later

def update(frame):
    global scatter_dots, trail_lines
    for dot in scatter_dots:
        dot.remove()
    for line in trail_lines:
        line.remove()
    
    scatter_dots = []
    trail_lines = []

    current_frame = df[df['Frame'] == frame]

    for uid, color in zip(unique_ids, colors):
        # Extract data for the current pedestrian ID
        ped_data = current_frame[current_frame['ID'] == uid]
        
        # If the pedestrian exists in the current frame
        if not ped_data.empty:
            x, y = ped_data['X'].values[0], ped_data['Y'].values[0]
            
            # Update trail dictionary
            if uid not in trail_dict:
                trail_dict[uid] = []
            trail_dict[uid].append((x, y))
            
            # Keep only the last 10 positions for the trail
            trail_dict[uid] = trail_dict[uid][-10:]
            
            # Draw trail line
            trail_x, trail_y = zip(*trail_dict[uid])
            line, = ax.plot(trail_x, trail_y, color=color)
            trail_lines.append(line)
            
            # Draw current position
            dot = ax.scatter(x, y, color=color, s=100)
            scatter_dots.append(dot)
            
            # Add ID number inside the circle
            text = ax.text(x, y, str(int(uid)), color='black', ha='center', va='center', fontweight='bold', fontsize=12)
            scatter_dots.append(text)
        
    
    # Draw virtual pedestrian
    virtual_ped = df_virtual[df_virtual['Frame'] == frame]
    
    if not virtual_ped.empty:
        x, y = virtual_ped['X'].values[0], virtual_ped['Y'].values[0]
        
        uid = len(unique_ids)
        if uid not in trail_dict:
            trail_dict[uid] = []
        trail_dict[uid].append((x, y))
        trail_dict[uid] = trail_dict[uid][-10:]
        trail_x, trail_y = zip(*trail_dict[uid])
        line, = ax.plot(trail_x, trail_y, color='black')
        trail_lines.append(line)
        
        dot = ax.scatter(x, y, color='black', s=100)
        scatter_dots.append(dot)
        cross = ax.scatter(virtual_ped['TargetX'].values[0], virtual_ped['TargetY'].values[0], color='black', marker='x', s=100)
        scatter_dots.append(cross)
    
    # Save current frame as png
    plt.savefig(f'img/{str(int(frame//10))}/frame_{frame}.png')

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=np.unique(df['Frame']), interval=200)

# Save as GIF
ani.save('gif/pedestrian_dynamics_with_virtual.gif', writer='pillow')
print('Animation saved!')
