import numpy as np
import pandas as pd
from collections import defaultdict


## STATE REPRESENTATION ##
class Environment:
    def __init__(self, pedestrian_data):
        self.pedestrian_data = pedestrian_data
        self.virtual_pedestrian = {'y': 0, 'x': -6, 'v': 0, 'v_angle': np.pi/2}
        self.target_position = {'y': 0, 'x': 6}
        self.current_frame = 1

    def get_state(self):
        """Extract state details for the current frame."""
        pedestrians_current_frame = self.pedestrian_data[self.current_frame]
        state = {
            'virtual': self.virtual_pedestrian,
            'real': pedestrians_current_frame,
            'target': self.target_position
        }
        return state

    def reset(self):
        """Reset environment to the initial state."""
        self.virtual_pedestrian = {'y': 0, 'x': -6, 'v': 0, 'v_angle': np.pi/2}
        self.current_frame = 1
        return self.get_state()


## ACTIONS ##
ACTIONS = [0, 1, 2, 3]

def take_action(state, action, delta_t=4/30, tau=0.5, vd_max=1.5, d_a=2):
    virtual = state['virtual'].copy()
    
    v = virtual['v']
    distance_to_target = np.sqrt((virtual['x'] - state['target']['x'])**2 + (virtual['y'] - state['target']['y'])**2)

    # Calculate acceleration or deceleration based on distance to target
    if distance_to_target <= d_a:
        acceleration = -v / tau
    else:
        acceleration = (vd_max - v) / tau

    # Apply action
    if action == 0:  # Turn left and move forward
        virtual['v_angle'] -= np.pi / 12  # turn left by 15 degrees
    elif action == 1:  # Turn right and move forward
        virtual['v_angle'] += np.pi / 12  # turn right by 15 degrees
    elif action == 2:  # Accelerate and move forward
        virtual['v'] = min(v + acceleration * delta_t, vd_max)
    elif action == 3:  # Decelerate and move forward
        virtual['v'] = max(v - acceleration * delta_t, 0)

    virtual['x'] += np.cos(virtual['v_angle']) * v * delta_t
    virtual['y'] += np.sin(virtual['v_angle']) * v * delta_t
        
    next_state = state.copy()
    next_state['virtual'] = virtual
    return next_state


## REWARDS ##
def compute_reward(state, next_state, prev_top_n_distances=None, pedestrian_radius=0.3, top_n=5):
    # Constants
    COLLISION_PENALTY = -50
    NEAR_MISS_PENALTY = -10
    TARGET_REWARD = 100
    GETTING_CLOSER_PENALTY = 10
    time_penalties = -2 * env.current_frame

    virtual = next_state['virtual']
    real_pedestrians = next_state['real']

    # Check collision with real pedestrians and compute distances
    collision_penalties = 0
    distances = []
    for ped in real_pedestrians:
        distance = np.sqrt((virtual['x'] - ped['x'])**2 + (virtual['y'] - ped['y'])**2)
        distances.append(distance)
        if distance < 2 * pedestrian_radius:  # A collision has occurred
            collision_penalties += COLLISION_PENALTY

    # Consider the top_n nearest pedestrians for near miss penalty
    distances.sort()
    top_n_distances = distances[:top_n]
    near_miss_penalties = sum([NEAR_MISS_PENALTY for d in top_n_distances if d < 3 * pedestrian_radius])

    # Penalty if getting closer to the previous top 5 nearest pedestrians
    getting_closer_penalties = 0
    if prev_top_n_distances:
        for old, new in zip(prev_top_n_distances, top_n_distances):
            if new > old:
                getting_closer_penalties += GETTING_CLOSER_PENALTY

    # Check if closer to the target than before
    old_distance = np.sqrt((state['virtual']['x'] - state['target']['x'])**2 + (state['virtual']['y'] - state['target']['y'])**2)
    new_distance = np.sqrt((virtual['x'] - state['target']['x'])**2 + (virtual['y'] - state['target']['y'])**2)
    
    target_reward = TARGET_REWARD if new_distance < old_distance else 0

    # Combine rewards
    total_reward = collision_penalties + near_miss_penalties + getting_closer_penalties + target_reward + time_penalties
    
    print(total_reward)
    return total_reward, top_n_distances


## Q-LEARNING ##
Q = defaultdict(float)
alpha = 0.1
gamma = 0.9
epsilon = 0.5
global prev_top_n_distances

def q_learning_step(env, state, Q):
    global prev_top_n_distances
    
    if env.current_frame == 1:
        prev_top_n_distances = None

    state['real'] = env.pedestrian_data[env.current_frame]

    state_str = str(state)
    if np.random.rand() < epsilon:
        action = np.random.choice(ACTIONS)
    else:
        action = max(ACTIONS, key=lambda x: Q[(state_str, x)])

    next_state = take_action(state, action)
    next_state['real'] = env.pedestrian_data[env.current_frame + 1]
    reward, prev_top_n_distances = compute_reward(state, next_state, prev_top_n_distances)

    next_state_str = str(next_state)
    best_next_action = max(ACTIONS, key=lambda x: Q[(next_state_str, x)])
    td_target = reward + gamma * Q[(next_state_str, best_next_action)]
    td_error = td_target - Q[(state_str, action)]
    Q[(state_str, action)] += alpha * td_error

    return next_state


## TRAIN ##
def has_reached_target(virtual_position, target_position, threshold=0.1):
    distance_to_target = np.sqrt((virtual_position['x'] - target_position['x'])**2 + (virtual_position['y'] - target_position['y'])**2)
    return distance_to_target < threshold
  
def train(env, Q, num_episodes=1000, max_frames=250):
    for _ in range(num_episodes):
        state = env.reset()
        while True:
            state = q_learning_step(env, state, Q)
            env.current_frame += 1
            
            if env.current_frame >= max_frames or has_reached_target(state['virtual'], state['target']):
                break


## OBSERVE ##
def observe_virtual_pedestrian(env, Q):
    state = env.reset()
    positions = []

    while not has_reached_target(state['virtual'], state['target']) and env.current_frame < 252:
        state_str = str(state)
        action = max(ACTIONS, key=lambda x: Q[(state_str, x)])
        state = take_action(state, action)
        positions.append((env.current_frame, state['virtual']['x'], state['virtual']['y']))
        env.current_frame += 1

    # Save positions to a txt file
    with open('../txt/virtual_pedestrian_trajectory.txt', 'w') as f:
        for frame, x, y in positions:
            f.write(f"{frame:.1f} {y:.7f} {x:.7f}\n")

    return positions


# Create the environment
df = pd.read_csv('../txt/merged_trajectories_with_velocity.txt', delim_whitespace=True, header=None, names=['frame', 'y', 'x', 'ID', 'velocity'])
env = Environment(df.groupby('frame').apply(lambda x: x.to_dict('records')).to_dict())

# Train the agent
train(env, Q, num_episodes=1000, max_frames=251)

# Observe the virtual pedestrian's path
observe_virtual_pedestrian(env, Q)