import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
import os


taus = [0.2 + i * 0.01 for i in range(int((3 - 0.2) / 0.01) + 1)]
# Time step between frames
dt = 4 / 30
# Read the merged txt file into a DataFrame
df = pd.read_csv('../../txt/merged_trajectories_with_velocity.txt', delim_whitespace=True, header=None, names=['Frame', 'Y', 'X', 'ID', 'Velocity'])

# Calculate velocity for one particle
def velocity_trajectory(uid):
    ped_data = df[df['ID'] == uid].sort_values('Frame')
    
def trajectory_graph(id, vd, frames):
    ped_data = df[df['ID'] == id].sort_values('Frame')
    plt.figure()
    plt.plot(ped_data['Frame']*4/30, ped_data['Velocity'], marker = "x")
    
    for frame_set in frames:
        for frame in frame_set:
            plt.axvline(x=frame *4/30, color='red', linestyle='--')

    plt.axhline(y=vd, color='green', linestyle='-.', label='vd')
    plt.legend()
    plt.savefig(f'../../img/ej2/{int(id)}_trajectory.png')
    plt.close()

def sfm_1D(tau, vd, v0, from_frame, to_frame, m = 70):
    """Simulates the Social Force Model in 1D for a single particle."""
    # Initialization
    num_steps = (to_frame - from_frame + 1 ) * 100
    v = np.zeros(num_steps)
    v[0] = v0

    # Simulation Loop
    for i in range(1, num_steps):
        # Compute the autopropulsion force
        f = m * (vd - v[i-1]) / tau
        # Compute the acceleration
        a = f / m
        # Update the velocity using the acceleration
        v[i] = v[i-1] + a * dt/100

    return v



#desac
def calcTauA(uid, vd, vmin, from_frame, to_frame):
    
    velocity_trajectory(uid)
    #get initial position
    row0 = df[df['Frame'] == from_frame]
    x0 = row0['X'].values[0]
    y0 = row0['Y'].values[0]
    #get final position
    rowf = df[df['Frame'] == to_frame]
    xf = rowf['X'].values[0]
    yf = rowf['Y'].values[0]
    #da
    da = math.sqrt((xf - x0)**2 + (yf - y0)**2)

    t = np.arange(from_frame, to_frame + 1, 1)
    t = t * 4 / 30

    original_trajectory = df[(df['ID'] == uid) & (df['Frame'] >= from_frame) & (df['Frame'] <= to_frame)]
    v_original = original_trajectory['Velocity']
    errors = []


    taus = [0.2 + i * 0.01 for i in range(int((3 - 0.2) / 0.01) + 1)]
    
    for i, tau in enumerate(taus):
        v = sfm_1D(tau, vmin, vd, from_frame, to_frame)
        v = v[::100]
        aux = np.mean((v_original - v) **2)
        errors.append((tau, aux))
    
    taus = [tau for tau, mse in errors]
    mse_values = [mse for tau, mse in errors]

    min_mse = min(mse_values)
    tauA = [tau for tau, mse in zip(taus, mse_values) if mse == min_mse][0]

    t = np.arange(from_frame, to_frame + 1, 1)
    t = t * 4 / 30
    plt.figure()
    plt.plot(original_trajectory['Frame']*4/30, original_trajectory['Velocity'], marker = "o", label='v-exp')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Velocidad $(\\frac{m}{s})$')
    v = sfm_1D(tauA, vmin, vd, from_frame, to_frame)
    v = v[::100]
    plt.plot(t, v, marker='o', label='v-sfm')
    plt.legend()
    plt.savefig(f'../../img/ej2/{int(uid)}_tauA_{tauA}.png')
    plt.close()

    return tauA, da


#ac
def calcTauP(uid, vd, vmin, from_frame, to_frame):
    velocity_trajectory(uid)

    t = np.arange(from_frame, to_frame + 1, 1)
    t = t * 4 / 30

    original_trajectory = df[(df['ID'] == uid) & (df['Frame'] >= from_frame) & (df['Frame'] <= to_frame)]
    v_original = original_trajectory['Velocity']
    errors = []

    
    taus = [0.2 + i * 0.01 for i in range(int((3 - 0.2) / 0.01) + 1)]

    for i, tau in enumerate(taus):
        v = sfm_1D(tau, vd, vmin, from_frame, to_frame)
        v = v[::100]
        aux = np.mean((v_original - v) **2)
        errors.append((tau, aux))
    
    taus = [tau for tau, mse in errors]
    mse_values = [mse for tau, mse in errors]

    min_mse = min(mse_values)
    tauP = [tau for tau, mse in zip(taus, mse_values) if mse == min_mse][0]

    t = np.arange(from_frame, to_frame + 1, 1)
    t = t * 4 / 30
    plt.figure()
    plt.plot(original_trajectory['Frame']*4/30, original_trajectory['Velocity'], marker = "o", label='v-exp')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Velocidad $(\\frac{m}{s})$')
    v = sfm_1D(tauP, vd, vmin,  from_frame, to_frame)
    v = v[::100]
    plt.plot(t, v, marker='o', label='v-sfm')
    plt.legend()
    plt.savefig(f'../../img/ej2/{int(uid)}_tauP_{tauP}.png')
    plt.close()

    return tauP

def eventTaus(id, vd, vmin, f1, f2, f3):
    ta, da = calcTauA(id, vd, vmin, f1, f2)
    tp = calcTauP(id, vd, vmin, f2, f3)
    return (da, ta, tp)


avg_dict = {}

def calc_avg(agent_id, vd, vmin, frames):
    data = []
    for vmin_value, frame_set in zip(vmin, frames):
        data.append(eventTaus(agent_id, vd, vmin_value, *frame_set))

    trajectory_graph(agent_id, vd, frames)
    print(data)
    # Avg
    avg_da = pd.Series([item[0] for item in data]).mean()
    avg_tauA = pd.Series([item[1] for item in data]).mean()
    avg_tauP = pd.Series([item[2] for item in data]).mean()

    avg_dict[agent_id] = {'vd': vd, 'Average_da': avg_da, 'Average_tauA': avg_tauA, 'Average_tauP': avg_tauP}


# Agent 1
vmins_1 = [0.3024966672957964, 0.0]
frames_agent_1 = [(55, 58, 63), (92, 98, 102)]
calc_avg(1, 1.25, vmins_1 ,frames_agent_1)
trajectory_graph(1,1.25,frames_agent_1)

# Agent 2
vmins_2 =[0.213895527771886, 0.2138947867901168]
frames_agent_2 = [(38, 46, 56), (153, 162, 176)]
calc_avg(2, 1.63, vmins_2, frames_agent_2)

# Agent 4
vmins_4= [0.0, 0.21389629455372]
frames_agent_4 = [(54, 63, 70), (180, 187, 193)]
calc_avg(4, 1.7, vmins_4, frames_agent_4)

# Agent 9
vmins_9 = [0.0,0.0]
frames_agent_9 = [(40, 51, 59), (86, 96, 106)]
calc_avg(9, 1.37, vmins_9, frames_agent_9)

# Agent 10
vmins_10=[0.21389629455372, 0.0, 0.0]
frames_agent_10 = [(64, 74, 83),(150, 159, 171), (199, 207, 214)]
calc_avg(10, 1.9, vmins_10, frames_agent_10)

# Agent 12
vmins_12 = [0.0, 0.0]
frames_agent_12 = [(106, 122, 133), (191, 208, 218)]
calc_avg(12, 1.5, vmins_12, frames_agent_12)

# Agent 15
vmins_15=[0.0, 0.0]
frames_agent_15 = [(141, 155, 168), (198, 209, 220)]
calc_avg(15, 1.9, vmins_15, frames_agent_15)

# Agent 18
vmins_18=[0.3024944985021428, 0.2138958365211782]
frames_agent_18 = [(100, 103, 106), (228, 231, 235)]
calc_avg(18, 1.5, vmins_18, frames_agent_18)

# Agent 21
vmins_21=[0.3024944985021384, 0.2138947779639499, 0.0]
frames_agent_21 = [(15, 25, 34),(87, 94, 103), (208, 216, 226)]
calc_avg(21, 1.55, vmins_21, frames_agent_21)



# Extract the data from avg_dict
vd_values = [avg_dict[agent_id]['vd'] for agent_id in avg_dict]
avg_da_values = [avg_dict[agent_id]['Average_da'] for agent_id in avg_dict]
avg_tauA_values = [avg_dict[agent_id]['Average_tauA'] for agent_id in avg_dict]
avg_tauP_values = [avg_dict[agent_id]['Average_tauP'] for agent_id in avg_dict]

# Calculate the average and standard deviation
average_vd = np.mean(vd_values)
average_avg_da = np.mean(avg_da_values)
average_avg_tauA = np.mean(avg_tauA_values)
average_avg_tauP = np.mean(avg_tauP_values)

std_vd = np.std(vd_values)
std_avg_da = np.std(avg_da_values)
std_avg_tauA = np.std(avg_tauA_values)
std_avg_tauP = np.std(avg_tauP_values)

# Extract tauP and vd values from avg_dict
tauP_values = [avg_dict[agent_id]['Average_tauP'] for agent_id in avg_dict]
vd_values = [avg_dict[agent_id]['vd'] for agent_id in avg_dict]

# Create the scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(vd_values, tauP_values,  marker='o', label='Agents', color='blue')
plt.title('tauP vs. vd')
plt.xlabel('vd')
plt.ylabel('tauP')
plt.grid(True)
plt.legend()
plt.show()

# Print the results
print(f"Average vd: {average_vd:.2f}, Std vd: {std_vd:.2f}")
print(f"Average Average_da: {average_avg_da:.2f}, Std Average_da: {std_avg_da:.2f}")
print(f"Average Average_tauA: {average_avg_tauA:.2f}, Std Average_tauA: {std_avg_tauA:.2f}")
print(f"Average Average_tauP: {average_avg_tauP:.2f}, Std Average_tauP: {std_avg_tauP:.2f}")


# dataframe
avg_df = pd.DataFrame(avg_dict).T
# print table
print(avg_df)


# da vs tauA
da_values = avg_df['Average_da']
tauA_values = avg_df['Average_tauA']

plt.figure(figsize=(10, 6))
plt.scatter(da_values, tauA_values, marker='o', label='Agentes')
plt.title('TauA en función de da')
plt.xlabel('da')
plt.ylabel('TauA')
plt.grid(True)
plt.legend()
plt.show()


avg_df = avg_df.round(2)

# add uid
avg_df['Agent'] = avg_df.index
promedios_df = avg_df[['Agent'] + [col for col in avg_df.columns if col != 'Agent']]

fig, ax = plt.subplots(figsize=(8, 6))

# create table
tabla = ax.table(cellText=promedios_df.values, colLabels=promedios_df.columns, cellLoc='center', loc='center', colColours=['#f0f0f0', '#f0f0f0', '#f0f0f0', '#f0f0f0'])

# table format
tabla.auto_set_font_size(False)
tabla.set_fontsize(12)
tabla.scale(1, 1.5)

# hide axis
ax.axis('off')

# save img
plt.savefig('tabla_promedios.png', bbox_inches='tight', pad_inches=0.1)
plt.show()
