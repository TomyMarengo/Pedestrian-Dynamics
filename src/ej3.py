import math
import pandas as pd
import numpy as np



# Read the merged txt file into a DataFrame
df = pd.read_csv('../txt/merged_trajectories_with_vx_vy.txt', delim_whitespace=True, header=None, names=['Frame', 'Y', 'X', 'ID', 'Velocity', 'vy', 'vx'])
dt = 4 / 30

def dist(x1, y1, x2, y2 ):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def mg(a, b):
    return math.sqrt((a**2)+(b**2))

targets= [(-9.75, 6.5), (-3.25, -6.5), (3.25, -6.5), (9.75, 6.5)]

frame = 1
x = 9.75
y = -6.5
v = 0
vd= 1.7
da= 1.1
ta = 0.63
tp = 0.66
tau = tp
vx =0
vy =0
target = targets[0]
i_target = 0

t_dist = dist(x,y, target[0], target[1])

dist_total = t_dist
for i in range(0, 2):
    dist_total = dist_total + dist(targets[i][0], targets[i][1], targets[i+1][0],targets[i+1][1])

print(dist_total)

#Get the time from frames
time = df['Frame'] * 4/30

# Get the velocity for each particle in for one frame
def particle_velocity(frame):
    if df['Frame'] == frame:
        particles = df.set_index('ID')['Velocity'].to_dict()
        return particles
    
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

def sfm_eulermod (tau, vdx, vdy, v0x, v0y, x0, y0, m=70 ):
    num_steps = 100
    x = np.zeros(num_steps)
    x[0] = x0
    y = np.zeros(num_steps)
    y[0] = y0
    vx = np.zeros(num_steps)
    vx[0] = v0x
    vy = np.zeros(num_steps)
    vy[0] = v0y

    step_dt = dt / num_steps
    for i in range(1, num_steps):
        f = m * (vdx - vx[i-1]) / tau
        a = f / m
        vx[i] = vx[i-1] + a * step_dt
        x[i] = x[i-1]+ vx[i]*step_dt +(a/2)*(step_dt**2)

        f = m * (vdy - vy[i-1]) / tau
        a = f / m
        vy[i] = vy[i-1] + a * step_dt
        y[i] = y[i-1]+ vy[i]*step_dt +(a/2)*(step_dt**2)
        

    return x[num_steps-1], vx[num_steps-1], y[num_steps-1], vy[num_steps-1]




def reach_target(i): 
    tau = tp
    target = targets[i+1] #aca iria el
    t_dist =dist(x,y, target[0], target[1])


def collides (x2, y2, vx2, vy2):
    deltaVx = vx2 -vx
    deltaVy = vy2 - vy
    deltaX = x2 -x
    deltaY = y2 - y

    sigma = "suma radios"
    deltaVR = deltaVx * deltaX + deltaVy *deltaY
    deltaVV = deltaVx*deltaVx + deltaVy*deltaVy
    deltaRR = deltaX * deltaX + deltaY * deltaY

    d = deltaVR * deltaVR - deltaVV * (deltaRR - sigma * sigma)


    if deltaVR >= 0 or d<0:
        tc = -1
    else: 
        tc = - (deltaVR + math.sqrt(d)) / deltaVV
    
    return tc


def next_collisions():
    filas = df.loc[df['Frame'] == frame]
    collisions = []
    for index, fila in filas.iterrows():
        id = fila['ID']
        x2 = fila['X']
        y2 = fila['Y']
        vx2 = fila['vx']
        vy2 = fila['vy']
        
        tc = collides(x2, y2, vx2, vy2)
        
        if tc > 0:
            collisions.append((id, tc))
    
    


def simulate():

    targets= [(-9.75, 6.5), (-3.25, -6.5), (3.25, -6.5), (9.75, 6.5)]

    frame = 1
    x = 9.75
    y = -6.5
    v = 0
    vd= 1.59
    da= 1.44
    ta = 0.95
    tp = 0.62
    tau = tp
    vx =0
    vy =0
    target = targets[0]
    i_target = 0

    t_dist = dist(x,y, target[0], target[1])
    trajectory = []
    trajectory.append((1, y,x, 0))

    

    while frame < 252 and i_target < len(targets):

        distX = target[0] - x
        distY = target[1] - y
        vdx = vd * (distX / mg(distX, distY))
        vdy = vd * (distY / mg(distX, distY))
        
        if tau ==tp:
            xn, vxn, yn, vyn = sfm_eulermod (tau, vdx, vdy, vx, vy, x, y, m=70)
        elif tau == ta:
            xn, vxn, yn, vyn = sfm_eulermod (tau, 0, 0, vx, vy, x, y, m=70)
        trajectory.append((frame, yn,xn, mg(vxn, vyn)))
        x = xn
        y=yn
        vx = vxn
        vy = vyn

        t_dist = dist(x,y, target[0], target[1])
        
        if t_dist < da :
            tau = ta
            print('da')
        if t_dist < 0.1:
            print('target')
            if i_target == len(targets)-1:
                break
            tau = tp
            target = targets[i_target+1]
            t_dist =dist(x,y, target[0], target[1])
            i_target = i_target + 1
                 
        frame = frame +1
            
        #next_collisions()

    trajectory = pd.DataFrame(trajectory, columns=['Frame', 'Y', 'X', 'Velocity'])
    
    return trajectory

    #empiezo
    #calculo colisiones pero al principio va a dar 0 las v asi q no se
    #ver si estoy a la distancia con la que tengo q cambiar el tau
    #veo los proximos choques o el proximo --> muevo el target si va a chocar
    #calculo de nuevo



print
df2 = simulate()

nombre_archivo = '../txt/virtual_pedestrian_trajectory.txt'
df2.to_csv(nombre_archivo, sep=' ', header=False, index=False)