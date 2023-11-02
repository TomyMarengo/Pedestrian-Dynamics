import math
import pandas as pd
import numpy as np

def dist(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2' """
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def beeman(r, v, f, last_f, e_target, tau, VD):
    # Constants
    dt = 4/30/100
    m = 70
    
    # Predict the position and velocity
    r_pred_x = r[0] + v[0] * dt + (2 / 3) * f[0] * (dt ** 2) / m - (1 / 6) * last_f[0] * (dt ** 2) / m
    r_pred_y = r[1] + v[1] * dt + (2 / 3) * f[1] * (dt ** 2) / m - (1 / 6) * last_f[1] * (dt ** 2) / m
    r_pred = (r_pred_x, r_pred_y)
    
    v_pred_x = v[0] + (3 / 2) * f[0] * dt / m - (1 / 2) * last_f[0] * dt / m
    v_pred_y = v[1] + (3 / 2) * f[1] * dt / m - (1 / 2) * last_f[1] * dt / m
    
    # Calculate the force at the predicted position
    f_pred_x = m * (VD * e_target[0] - v_pred_x) / tau
    f_pred_y = m * (VD * e_target[1] - v_pred_y) / tau
    
    # Correct the velocity
    v_corr_x = v[0] + (1 / 3) * f_pred_x * dt / m + (5 / 6) * f[0] * dt / m - (1 / 6) * last_f[0] * dt / m 
    v_corr_y = v[1] + (1 / 3) * f_pred_y * dt / m + (5 / 6) * f[1] * dt / m - (1 / 6) * last_f[1] * dt / m
    v_corr = (v_corr_x, v_corr_y)
    
    return r_pred, v_corr

class VirtualPedestrian:
    def __init__(self, initial_position, targets, VD, DA, TA, TP, df):
        self.VD = VD
        self.DA = DA
        self.TA = TA
        self.TP = TP
        self.MASS = 70
        self.position = initial_position
        self.v = (0, 0)
        self.force = (0, 0)
        self.last_force = self.force

        self.targets = targets
        self.i_target = 0
        self.target = self.targets[0]
        self.calculate_e_target()

        self.frame = 1
        self.df = df
        self.adjustment_factor = 0.90

    def calculate_e_target(self):
        dx = self.target[0] - self.position[0]
        dy = self.target[1] - self.position[1]
        norm = math.sqrt(dx ** 2 + dy ** 2)
        self.e_target = (dx / norm, dy / norm)
    
    def avoid_collision(self):
        frame_df = self.df.loc[self.df['Frame'] == self.frame]
        min_distance = float('inf')
        closest_pedestrian = None
        
        for i in range(len(frame_df)):
            distance = dist(frame_df.iloc[i]['X'], frame_df.iloc[i]['Y'], self.position[0], self.position[1])
            if distance < min_distance:
                min_distance = distance
                closest_pedestrian = frame_df.iloc[i]
        
        if min_distance < 0.6:  # Assuming 0.6 is the collision threshold
            # Get directional vector of the closest pedestrian
            ped_direction = (closest_pedestrian['vx'], closest_pedestrian['vy'])
            
            # Get directional vector towards the real target
            dx = self.target[0] - self.position[0]
            dy = self.target[1] - self.position[1]
            my_direction = (dx, dy)
            
            # Calculate temporary target based on the sum of vectors
            temp_dx = ped_direction[0] + my_direction[0]
            temp_dy = ped_direction[1] + my_direction[1]
            
            # Update the temporary target
            self.target = (self.position[0] + temp_dx, self.position[1] + temp_dy)
        else:
            self.target = self.targets[self.i_target]
        self.calculate_e_target()

    def heading_to_same_target(self):
        frame_df = self.df.loc[self.df['Frame'] == self.frame]
        vector_to_temp_target = np.array(self.target) - np.array(self.position)
        distance_to_temp_target = np.linalg.norm(vector_to_temp_target)

        for _, pedestrian in frame_df.iterrows():
            # Pedestrian's velocity vector
            pedestrian_velocity_vector = np.array([pedestrian['vx'], pedestrian['vy']])
            pedestrian_pos = np.array([pedestrian['X'], pedestrian['Y']])
            pedestrian_to_temp_target_vector = np.array(self.target) - pedestrian_pos
            
            # Check if the pedestrian is moving
            if np.linalg.norm(pedestrian_velocity_vector) > 0:
                # Angle between the direction of the real pedestrian and the virtual pedestrian's temp target
                angle_to_target = angle_between(pedestrian_velocity_vector, pedestrian_to_temp_target_vector)
                
                # Convert angle to degrees for comparison
                angle_to_target_deg = np.degrees(angle_to_target)
                
                # Check if the real pedestrian is heading towards the virtual pedestrian's target within a 7-degree margin
                if abs(angle_to_target_deg) <= 7:
                    # If within distance DA, decide whether to slow down or speed up based on who is closer
                    if distance_to_temp_target <= self.DA:
                        self.v = (max(0, self.v[0] * self.adjustment_factor), max(0, self.v[1] * self.adjustment_factor))  # Slow down
                    else:
                        self.v = (min(self.e_target[0] * self.VD, self.v[0] + (self.v[0] * (1 - self.adjustment_factor))),
                                min(self.e_target[1] * self.VD, self.v[1] + (self.v[1] * (1 - self.adjustment_factor))))  # Speed up
                    return True
        return False

    def calculate_collisions(self):
        frame_df = self.df.loc[self.df['Frame'] == self.frame]
        for i in range(len(frame_df)):
            if dist(frame_df.iloc[i]['X'], frame_df.iloc[i]['Y'], self.position[0], self.position[1]) < 0.6:  # 2 * 0.3 radius
                pass

    def calculate_new_position(self):
        # self.calculate_collisions()
        # self.avoid_collision()
        # self.heading_to_same_target()
        
        self.calculate_e_target()
        # Calculate distance to the target
        distance_to_target = dist(self.position[0], self.position[1], self.target[0], self.target[1])
        # Choose the appropriate tau
        tau = self.TA if distance_to_target < self.DA else self.TP

        # Update velocity using the social force model: F = m * (vd * e - v) / tau
        self.last_force = self.force
        
        force_x = self.MASS * (self.VD * self.e_target[0] - self.v[0]) / tau
        force_y = self.MASS * (self.VD * self.e_target[1] - self.v[1]) / tau
        self.force = (force_x, force_y)
        
        self.position, self.v = beeman(self.position, self.v, self.force, self.last_force, self.e_target, tau, self.VD)
        
        distance_to_target = dist(self.position[0], self.position[1], self.target[0], self.target[1])
        if distance_to_target < 0.2:
            self.i_target = (self.i_target + 1) % len(self.targets)
            self.target = self.targets[self.i_target]

        # Increment frame counter
        self.frame += 1
        
        # Write data to output file
        if self.frame % 100 == 0:
            with open('../../txt/virtual_pedestrian_trajectory.txt', 'a') as f:
                f.write(f"{self.frame//100 + 1}\t{self.position[1]}\t{self.position[0]}\t{self.v[0]}\t{self.v[1]}\t{self.target[1]}\t{self.target[0]}\n")


# Read the merged txt file into a DataFrame
df = pd.read_csv('../../txt/merged_trajectories_with_vx_vy.txt', delim_whitespace=True, header=None, names=['Frame', 'Y', 'X', 'ID', 'Velocity', 'vy', 'vx'])
targets = [(-9.75, 6.5), (-3.25, -6.5), (3.25, -6.5), (9.75, 6.5)]
initial_position = (9.75, -6.5)
VD = 1.59
DA = 1.44
TA = 0.95
TP = 0.62

with open('../../txt/virtual_pedestrian_trajectory.txt', 'a') as f:
    f.write(f"{1}\t{initial_position[1]}\t{initial_position[0]}\t{0}\t{0}\t{targets[0][1]}\t{targets[0][0]}\n")

pedestrian = VirtualPedestrian(initial_position, targets, VD, DA, TA, TP, df)
for i in range(25100):
    pedestrian.calculate_new_position()