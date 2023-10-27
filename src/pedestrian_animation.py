import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import numpy as np
import os

# Create img/ folder if not exists
if not os.path.exists('../gif'):
    os.makedirs('../gif')
    
# Define a list of unique colors
unique_colors = [
    'b', 'g', 'r', 'c', 'm', 'y', 'k',
    '#FFA500', '#FF4500', '#2E8B57', '#ADFF2F',
    '#FFD700', '#4B0082', '#7CFC00', '#00FA9A',
    '#BDB76B', '#6A5ACD', '#708090', '#4682B4',
    '#D2691E', '#DC143C', '#8B0000', '#2F4F4F',
    '#FF00FF', '#1E90FF'
]

# Read the filtered_merged_file into a DataFrame
df = pd.read_csv('..txt/merged_trajectories_with_velocity.txt', delim_whitespace=True, header=None, names=['Frame', 'Y', 'X', 'ID', 'Velocity'])

# Create a list of unique IDs and assign a unique color to each ID
unique_ids = df['ID'].unique()
colors = unique_colors

# Create a dictionary to keep track of trails
trail_dict = {}

# Initialize the plot
plt.figure(figsize=(8, 8), facecolor='none')
plt.axis('off')  # Desactiva los ejes
plt.xticks([])    # Elimina las marcas en el eje x
plt.yticks([]) 

fig, ax = plt.subplots()
scatter_dots = []
ax.set_axis_off()

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
            ax.set_axis_off()
            dot = ax.scatter(x, y, color=color, s=100)
            scatter_dots.append(dot)
            plt.savefig(f'..frames/frame_{int(frame)}.png', transparent=True)

    

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=np.unique(df['Frame']), interval=200)

# Save as GIF
ani.save('../gif/pedestrian_dynamics.gif', writer='pillow')
print('Animation saved!')
