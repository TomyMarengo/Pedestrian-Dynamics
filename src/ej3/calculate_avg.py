import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
import os

# Time step between frames
dt = 4 / 30
# Read the merged txt file into a DataFrame
df = pd.read_csv('../../txt/merged_trajectories_with_velocity.txt', delim_whitespace=True, header=None, names=['Frame', 'Y', 'X', 'ID', 'Velocity'])

def sfm_1D(tau, vd, v0, from_frame, to_frame, m = 70):
    num_steps = (to_frame - from_frame + 1 ) * 100
    v = np.zeros(num_steps)
    v[0] = v0

    # Simulation Loop
    for i in range(1, num_steps):
        f = m * (vd - v[i-1]) / tau
        a = f / m
        v[i] = v[i-1] + a * dt/100

    return v

#desac
def calcTauA(uid, vd, vmin, from_frame, to_frame):
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

    return tauA, da

def calcTauP(uid, vd, vmin, from_frame, to_frame):
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
    v = sfm_1D(tauP, vd, vmin,  from_frame, to_frame)
    v = v[::100]

    return tauP

def calcVd(uid):
    velocities = df[df['ID'] == uid]['Velocity'] 
    #la velocidad deseada de cada particula siempre es mayor a 1
    filtered_velocities = velocities[velocities > 1]
    vd = np.mean(filtered_velocities)
    print(str(uid) + " : " + str(vd))

    return vd

def calcVmin(uid):
    velocities = df[df['ID'] == uid]['Velocity'] 
    vmin = min(velocities)
    print(str(uid) + " : " + str(vmin))

    return vmin

# Define the parameters for the simulation
from_frame = 1
to_frame = 250

tau_values = []
da_values = []
tap_values = []

# Calculate the optimal tau value
for uid in df['ID']:
    #Calculate the vd
    vd = calcVd(uid)

    vmin = calcVmin(uid)

    # Calculate the optimal tau value and da for the current UID
    #quiero que sea de todos los frames, ahi deberia analizar por tramos
    tauA, da = calcTauA(uid, vd, vmin, from_frame, to_frame)
    tauP = calcTauP(uid, vd, vmin, from_frame, to_frame)
    
    # Append the results to the lists
    tau_values.append(tauA)
    tau_values.append(tauP)
    da_values.append(da)

    print(uid)
    print("Optimal tau:", tauA)
    print("Distance traveled (da):", da)

