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

#Constant data of the equations
Aa = 1,25 
Ba = 1,25

#the position is updated
#vi = eit * v
#xi = xi + vi*t

#the direction eit 
#eit = xv - xi / abs(xv - xi) 

#the overlap
#eij = xi - xj / abs(xi - xj) 

#collision vector
#nc = eij * Aa * math.exp(-dij / (Ba * math.cos(Oj)))

#avoidance direction
#eia = nc + eit / abs(nc + eit)                     