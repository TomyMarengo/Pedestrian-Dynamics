import math
import pandas as pd
import numpy as np
from numpy import cross, array

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
    def __init__(self, initial_position, targets, VD, DA, TA, TP, df, collisions, distance_travelled, minimum_distances):
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
        self.real_frame = 1
        self.df = df
        self.adjustment_factor = 0.90
        
        self.collisions = collisions # DATA
        self.distance_travelled = distance_travelled # DATA
        self.minimum_distances = minimum_distances # DATA

    def calculate_e_target(self):
        dx = self.target[0] - self.position[0]
        dy = self.target[1] - self.position[1]
        norm = math.sqrt(dx ** 2 + dy ** 2)
        self.e_target = (dx / norm, dy / norm)
    
    def calculate_e_real_target(self):
        dx = self.targets[self.i_target][0] - self.position[0]
        dy = self.targets[self.i_target][1] - self.position[1]
        norm = math.sqrt(dx ** 2 + dy ** 2)
        self.e_target = (dx / norm, dy / norm)
    
    def avoid_collision(self):
        frame_df = self.df.loc[self.df['Frame'] == self.real_frame]
        min_distance = float('inf')
        closest_pedestrian = None

        for _, pedestrian in frame_df.iterrows():
            distance = dist(pedestrian['X'], pedestrian['Y'], self.position[0], self.position[1])
            if distance < min_distance:
                min_distance = distance
                closest_pedestrian = pedestrian
        
        # DATA       
        self.minimum_distances.append(min_distance) 
        if min_distance < 0.6:
            self.collisions.append((self.real_frame, closest_pedestrian['ID']))
        
        if min_distance < 1.2 and closest_pedestrian['vx'] * closest_pedestrian['vy'] != 0:  # 1 more than 0.6 of the collision radius
            # Get directional vector of the closest pedestrian
            ped_direction = (closest_pedestrian['vx'], closest_pedestrian['vy'])
            # Normalize the vector
            norm = math.sqrt(ped_direction[0] ** 2 + ped_direction[1] ** 2)
            ped_direction = (ped_direction[0] / norm, ped_direction[1] / norm)
            # Get directional vector of the virtual pedestrian
            self.calculate_e_real_target()
            my_direction = self.e_target
            # Calculate the angle between the two vectors
            dot_product = np.dot(ped_direction, my_direction)
            angle = np.arccos(dot_product / (np.linalg.norm(ped_direction) * np.linalg.norm(my_direction)))
            angle_degrees = np.degrees(angle)
            
            cross_prod = cross(array(my_direction), array(ped_direction))

            if angle_degrees < 60:  # Pedestrian is behind me
                # Choose direction based on the cross product
                if cross_prod > 0:  # Pedestrian is on my left
                    temp_direction = (-ped_direction[1], ped_direction[0])  # Rotate clockwise
                else:  # Pedestrian is on my right
                    temp_direction = (ped_direction[1], -ped_direction[0])  # Rotate counter-clockwise
            elif angle_degrees > 120:  # Pedestrian is in front of me
                # Similar logic for the pedestrian in front
                if cross_prod < 0:
                    temp_direction = (-ped_direction[1], ped_direction[0])
                else:
                    temp_direction = (ped_direction[1], -ped_direction[0])
            else:
                # Pedestrian is on the side, combine vectors
                temp_dx = ped_direction[0] + my_direction[0]
                temp_dy = ped_direction[1] + my_direction[1]
                temp_direction = (temp_dx, temp_dy)
            
            # Normalize the temporary direction
            norm_temp = math.sqrt(temp_direction[0] ** 2 + temp_direction[1] ** 2)
            temp_direction_normalized = (temp_direction[0] / norm_temp, temp_direction[1] / norm_temp)
    
            # Update the temporary target
            self.target = temp_direction_normalized
        else:
            self.target = self.targets[self.i_target]

    def heading_to_same_target(self):
        frame_df = self.df.loc[self.df['Frame'] == self.real_frame]
        self.calculate_e_target()
        distance_to_temp_target = dist(self.position[0], self.position[1], self.target[0], self.target[1])
        
        min_distance = float('inf')
        closest_pedestrian = None
        
        for _, pedestrian in frame_df.iterrows():
            distance = dist(pedestrian['X'], pedestrian['Y'], self.position[0], self.position[1])
            if distance < min_distance:
                min_distance = distance
                closest_pedestrian = pedestrian

        v1 = (closest_pedestrian['vx'], closest_pedestrian['vy'])
        v2 = (self.target[0] - closest_pedestrian['X'], self.target[1] - closest_pedestrian['Y'])
        pedestrian_distance_to_temp_target = dist(self.position[0], self.position[1], closest_pedestrian['X'], closest_pedestrian['Y'])
        
        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        # Magnitudes
        magnitude_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
        magnitude_v2 = math.sqrt(v2[0]**2 + v2[1]**2)
        # Angle in radians
        # To avoid division by zero or math domain errors due to floating point arithmetic,
        # make sure the denominator is not zero and the value inside arccos is within the valid range [-1, 1].
        cos_angle = dot_product / (magnitude_v1 * magnitude_v2) if magnitude_v1 * magnitude_v2 != 0 else 1
        cos_angle = min(1, max(-1, cos_angle))  # Clamp the value to the range [-1, 1]
        angle = math.acos(cos_angle)
        # Convert to degrees, if desired
        angle_degrees = math.degrees(angle)
        
        # print("Frame: ", self.real_frame)  
        # print("My distance to target: ", distance_to_temp_target)
        # print("Pedestrian distance to target: ", pedestrian_distance_to_temp_target)
        
        # Check if the real pedestrian is heading towards the virtual pedestrian's target within a 7-degree margin
        if (abs(angle_degrees) <= 10 and abs(angle_degrees) >= 0) or (abs(angle_degrees) >= 350 and abs(angle_degrees) <= 360):
            # If within distance DA, decide whether to slow down or speed up based on who is closer
            if pedestrian_distance_to_temp_target <= distance_to_temp_target <= self.DA:
                self.v = (0, 0)  # Slow down
                print("Stopped in ", self.real_frame)

    def calculate_collisions(self):
        frame_df = self.df.loc[self.df['Frame'] == self.real_frame]
        for i in range(len(frame_df)):
            if dist(frame_df.iloc[i]['X'], frame_df.iloc[i]['Y'], self.position[0], self.position[1]) < 0.6:  # 2 * 0.3 radius
                pass

    def calculate_new_position(self):
        self.last_force = self.force
        self.force = (0, 0)
        
        # self.calculate_collisions()
        self.avoid_collision()
        self.heading_to_same_target()
        
        # Calculate distance to the target
        distance_to_target = dist(self.position[0], self.position[1], self.targets[self.i_target][0], self.targets[self.i_target][1])
        # Choose the appropriate tau
        tau = self.TA if distance_to_target < self.DA else self.TP

        # Update velocity using the social force model: F = m * (vd * e - v) / tau
        self.calculate_e_target()
        force_x = self.MASS * (self.VD * self.e_target[0] - self.v[0]) / tau
        force_y = self.MASS * (self.VD * self.e_target[1] - self.v[1]) / tau
        self.force = (self.force[0] + force_x, self.force[1] + force_y)
        
        aux_position = self.position # DATA
        self.position, self.v = beeman(self.position, self.v, self.force, self.last_force, self.e_target, tau, self.VD)
        self.distance_travelled += dist(aux_position[0], aux_position[1], self.position[0], self.position[1]) # DATA
        
        distance_to_target = dist(self.position[0], self.position[1], self.targets[self.i_target][0], self.targets[self.i_target][1])
        if distance_to_target <= 0.3:
            self.i_target = (self.i_target + 1) % len(self.targets)
            self.target = self.targets[self.i_target]

        # Increment frame counter
        self.frame += 1
        
        # Write data to output file
        if self.frame % 100 == 0:
            self.real_frame += 1
            with open('../../txt/virtual_pedestrian_trajectory.txt', 'a') as f:
                f.write(f"{self.real_frame}\t{self.position[1]}\t{self.position[0]}\t{self.v[1]}\t{self.v[0]}\t{self.target[1]}\t{self.target[0]}\n")


# Read the merged txt file into a DataFrame
df = pd.read_csv('../../txt/merged_trajectories_with_vx_vy.txt', delim_whitespace=True, header=None, names=['Frame', 'Y', 'X', 'ID', 'Velocity', 'vy', 'vx'])
targets = [(-9.75, 6.5), (-3.25, -6.5), (3.25, -6.5), (9.75, 6.5)]
initial_position = (9.75, -6.5)
VD = 1.59
DA = 1.44
TA = 0.95
TP = 0.62

with open('../../txt/virtual_pedestrian_trajectory.txt', 'w') as f:
    f.write(f"{1}\t{initial_position[1]}\t{initial_position[0]}\t{0}\t{0}\t{targets[0][1]}\t{targets[0][0]}\n")

# DATA
collisions = []
distance_travelled = 0
minimum_distances = []

pedestrian = VirtualPedestrian(initial_position, targets, VD, DA, TA, TP, df, collisions, distance_travelled, minimum_distances)
for i in range(25000):
    pedestrian.calculate_new_position()

unique_collisions = sorted(list(set(pedestrian.collisions)), key=lambda x: x[0])
print(f"Minimum distances: {[pedestrian.minimum_distances[i] for i in range(len(pedestrian.minimum_distances)) if i % 100 == 0]} m")
print(f"Number of collisions: {len(unique_collisions)}")
print(f"Collisions: {unique_collisions}")
print(f"Distance travelled: {pedestrian.distance_travelled} m")
print(f"Average velocity: {pedestrian.distance_travelled / (25000 * 4/30/100)} m/s")