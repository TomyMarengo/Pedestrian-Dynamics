import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt

## PLOTS ##
rewards = []

# Parameters for your 2D space and blocks
width = 0.1
x_range = np.arange(-25, 25.01, width)
y_range = np.arange(-15, 15.01, width)

# Preallocate matrices
reward_accumulator = np.zeros((x_range.shape[0]-1, y_range.shape[0]-1))
count_matrix = np.zeros_like(reward_accumulator)

def get_block_idx(x, y, width):
    """Returns the indices of the block for given x and y coordinates"""
    x_idx = int(x // width)
    y_idx = int(y // width)
    return x_idx, y_idx
  
###########################################################################


## STATE REPRESENTATION ##
class Environment:
    def __init__(self, pedestrian_data):
        self.pedestrian_data = pedestrian_data
        self.virtual_pedestrian = {'y': 0, 'x': -6, 'v': 0, 'v_angle': 0}
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
        self.virtual_pedestrian = {'y': 0, 'x': -6, 'v': 0, 'v_angle': 0}
        self.current_frame = 1
        return self.get_state()


## ACTIONS ##
ACTIONS = [0, 1, 2, 3, 4]

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
        virtual['v_angle'] += 0  # don't turn
    elif action == 1:  # Turn right and move forward
        virtual['v_angle'] += np.pi / 6  # turn right by 15 degrees
    elif action == 2:  # Accelerate and move forward
        virtual['v_angle'] -= np.pi / 6  # turn left by 15 degrees
    elif action == 3:  # Turn right and move forward
        virtual['v_angle'] += np.pi / 3  # turn right by 15 degrees
    elif action == 4:  # Accelerate and move forward
        virtual['v_angle'] -= np.pi / 3  # turn left by 15 degrees
        
    virtual['v'] = max(0, min(v + acceleration * delta_t, vd_max))
    virtual['x'] += np.cos(virtual['v_angle']) * v * delta_t
    virtual['y'] += np.sin(virtual['v_angle']) * v * delta_t
        
    next_state = state.copy()
    next_state['virtual'] = virtual
    return next_state


## REWARDS ##
def compute_reward(state, next_state, action, pedestrian_radius=0.3):
    # Constants
    COLLISION_REWARD = -1000
    NEAR_REWARD = -100
    TARGET_REWARD = 200
    total_reward = -0.5 * env.current_frame

    virtual = next_state['virtual']
    real_pedestrians = next_state['real']

    distances = []
    for ped in real_pedestrians:
        distance = np.sqrt((virtual['x'] - ped['x'])**2 + (virtual['y'] - ped['y'])**2)
        if distance < 3.3 * pedestrian_radius:
            total_reward += NEAR_REWARD
        if distance < 2 * pedestrian_radius:
            total_reward += COLLISION_REWARD

    # Check if closer to the target than before
    old_distance = np.sqrt((state['virtual']['x'] - state['target']['x'])**2 + (state['virtual']['y'] - state['target']['y'])**2)
    new_distance = np.sqrt((virtual['x'] - state['target']['x'])**2 + (virtual['y'] - state['target']['y'])**2)
    
    if action in [3, 4]:
        total_reward -= 4 ** 4
    if action in [1, 2]:
        total_reward -= 2 ** 4

    if -15 > virtual['x'] > 15 or -7 > virtual['y'] > 7:
        total_reward += COLLISION_REWARD * 100

    if new_distance < old_distance:
        total_reward += 1 / (old_distance - new_distance)
        print(1 / (old_distance - new_distance))

    if new_distance < 1:
        total_reward += TARGET_REWARD * 3 / new_distance
        print("REACHEEEEEEEEEEEEEEEEEED!")
    elif new_distance < 2:
        total_reward += TARGET_REWARD * 2
    elif new_distance < 3:
        total_reward += TARGET_REWARD

    return total_reward


## Q-LEARNING ##
Q = defaultdict(lambda: np.random.uniform(0, 0.01))
alpha = 0.001
gamma = 0.99
global epsilon
epsilon_decay = 0.998

def q_learning_step(env, state, Q):
    global epsilon
    
    state['real'] = env.pedestrian_data[env.current_frame]

    state_str = str(state)
    if np.random.rand() < epsilon:
        action = np.random.choice(ACTIONS)
    else:
        action = max(ACTIONS, key=lambda x: Q[(state_str, x)])

    next_state = take_action(state, action)
    next_state['real'] = env.pedestrian_data[env.current_frame + 1]
    reward = compute_reward(state, next_state, action)
    rewards.append(reward)

    next_state_str = str(next_state)
    best_next_action = max(ACTIONS, key=lambda x: Q[(next_state_str, x)])
    td_target = reward + gamma * Q[(next_state_str, best_next_action)]
    td_error = td_target - Q[(state_str, action)]
    Q[(state_str, action)] += alpha * td_error

    return next_state


## TRAIN ##
def has_reached_target(virtual_position, target_position, threshold=0.5):
    distance_to_target = np.sqrt((virtual_position['x'] - target_position['x'])**2 + (virtual_position['y'] - target_position['y'])**2)
    return distance_to_target < threshold
  
def train(env, Q, num_episodes, max_frames):
    global epsilon
    
    epsilon = 1
    for _ in range(num_episodes):
        state = env.reset()
        epsilon = max(0.05, epsilon*epsilon_decay)
        while True:
            state = q_learning_step(env, state, Q)
            
            # Get block indices
            x_idx = np.digitize(state['virtual']['x'], x_range) - 1
            y_idx = np.digitize(state['virtual']['y'], y_range) - 1
            
            if x_idx > 0 and x_idx <= x_range.shape[0]-2 and y_idx > 0 and y_idx <= y_range.shape[0]-2:
                reward_accumulator[x_idx, y_idx] += rewards[-1]
                count_matrix[x_idx, y_idx] += 1
            
            env.current_frame += 1
            
            if env.current_frame >= max_frames or has_reached_target(state['virtual'], state['target']):
                break


## OBSERVE ##
def observe_virtual_pedestrian(env, Q):
    state = env.reset()
    positions = []
    positions.append((env.current_frame, state['virtual']['x'], state['virtual']['y'], state['virtual']['v'], None))
    
    while not has_reached_target(state['virtual'], state['target']) and env.current_frame < 252:
        state_str = str(state)
        action = max(ACTIONS, key=lambda x: Q[(state_str, x)])
        state = take_action(state, action)
        env.current_frame += 1
        positions.append((env.current_frame, state['virtual']['x'], state['virtual']['y'], state['virtual']['v'], action))

    # Save positions to a txt file
    with open('../txt/virtual_pedestrian_trajectory.txt', 'w') as f:
        for frame, x, y, v, action in positions:
            f.write(f"{frame:.1f} {y:.7f} {x:.7f} {v:.7f} {action}\n")

    return positions


## RUN ##
# Create the environment
df = pd.read_csv('../txt/merged_trajectories_with_velocity.txt', delim_whitespace=True, header=None, names=['frame', 'y', 'x', 'ID', 'velocity'])
env = Environment(df.groupby('frame').apply(lambda x: x.to_dict('records')).to_dict())

# Train the agent
train(env, Q, num_episodes=2000, max_frames=251)

# Observe the virtual pedestrian's path
observe_virtual_pedestrian(env, Q)


## PLOTS ##

# 2D
plt.plot(rewards, marker='o', markersize=1, linestyle='none', color='blue')
plt.xlabel('Episodes')
plt.ylabel('Total Reward')
plt.show()
plt.savefig('../img/reward_2d.png')

# 3D

# Compute the average reward for each block, handling blocks with zero count
average_rewards = np.divide(reward_accumulator, count_matrix, out=np.zeros_like(reward_accumulator), where=count_matrix!=0)

# Create grid
X, Y = np.meshgrid(x_range[:-1], y_range[:-1])

# Plotting
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(X, Y, average_rewards.T, cmap='viridis')  # transpose for correct orientation

ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Average Reward')
ax.set_title('Average Reward in Virtual Pedestrian Space')
fig.colorbar(surf)
plt.show()
plt.savefig('../img/reward_space.png')


