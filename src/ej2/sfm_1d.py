import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# CHANGE UID HERE
uid = 2
# CHANGE PARAMETERS HERE
taus = [0.1+ i * 0.01 for i in range(int((3-0.3) / 0.01) + 1)]
from_frame = 38
to_frame = 46
vd = 1.63
# Time step between frames
dt = 4 / 30
# Read the merged txt file into a DataFrame
df = pd.read_csv('../../txt/merged_trajectories_with_velocity.txt', delim_whitespace=True, header=None, names=['Frame', 'Y', 'X', 'ID', 'Velocity'])

# Create img/ folder if not exists
if not os.path.exists('img'):
    os.makedirs('img')

# Calculate velocity for one particle
def velocity_trajectory(uid):
    ped_data = df[df['ID'] == uid].sort_values('Frame')
    plt.figure()
    plt.plot(ped_data['Frame']*4/30, ped_data['Velocity'], marker = "x")

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

# GRAPH PLOTS
velocity_trajectory(uid)

# Plotting SFM with different tau values
t = np.arange(from_frame, to_frame + 1, 1)
t = t * 4 / 30

color_map = plt.get_cmap('viridis')

original_trajectory = df[(df['ID'] == uid) & (df['Frame'] >= from_frame) & (df['Frame'] <= to_frame)]
v_original = original_trajectory['Velocity']
errors = []

for i, tau in enumerate(taus):
    v = sfm_1D(tau, 0.25, vd,   from_frame, to_frame)
    v = v[::100]
    aux = np.mean((v_original - v) **2)
    errors.append((tau, aux))
    colorr = color_map(i / len(taus))
    plt.plot(t, v, color=colorr, marker='o', markersize=4 , linestyle='None')


plt.xlabel('Tiempo (s)')
plt.ylabel('Velocidad $(\\frac{m}{s})$')
plt.title(f'Agente {int(uid)}')  # TODO: Quitar esto
plt.legend()
plt.savefig(f'img/velocity_tau_{int(uid)}.png')
plt.show()
print(f'Graph saved at img/velocity_tau_{int(uid)}.png')

# Extraer los valores de tau y MSE
taus = [tau for tau, mse in errors]
mse_values = [mse for tau, mse in errors]

min_mse = min(mse_values)
tau_for_min_mse = [tau for tau, mse in zip(taus, mse_values) if mse == min_mse][0]
print(tau_for_min_mse)

# Crear un gráfico de puntos con línea
plt.plot(taus, mse_values, marker='o', linestyle='-')
plt.xlabel('Valor de Tau')
plt.ylabel('Error Cuadrático Medio (MSE)')
plt.title('MSE vs Tau')
plt.grid(True)
plt.show()

plt.figure()
plt.plot(original_trajectory['Frame']*4/30, original_trajectory['Velocity'], marker = "o")
v = sfm_1D(tau_for_min_mse, 0.25, vd, from_frame, to_frame)
v = v[::100]
plt.plot(t, v, marker='o')
plt.show()