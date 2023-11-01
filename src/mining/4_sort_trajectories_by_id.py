import pandas as pd

# Leer el archivo TXT en un DataFrame
df = pd.read_csv('../../txt/merged_trajectories_with_velocity.txt', delim_whitespace=True, header=None, names=['Frame', 'Y', 'X', 'ID', 'Velocity'])

# Ordenar los datos por ID
df_sorted = df.sort_values(by=['ID', 'Frame'])

# Guardar el nuevo DataFrame ordenado en un nuevo archivo TXT
df_sorted.to_csv('../../txt/merged_trajectories_with_velocity_sorted.txt', sep=' ', index=False, header=False)

print('Done!')
