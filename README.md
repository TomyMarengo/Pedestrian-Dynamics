# Pedestrian Dynamics: University Practical Work

## Overview

This practical work for university aims to study pedestrian dynamics through simulation data. We utilize Python as our primary language for implementation, with the use of libraries such as Pandas for data manipulation, Matplotlib for plotting, and NumPy for numerical computations.

## Objectives

1. **Data Preprocessing**: Merging two text files that contain the position (X, Y) and ID of different particles (pedestrians) at various frames. One file contained particles with IDs from 0 to 13, and the other contained particles with IDs initially from 0 to 10. We adjusted the IDs of the second file to continue from the first, making them 14 to 25. 
2. **Data Filtering**: Remove all data entries with a frame number greater or equal to 252 because the simulation was stopped at that point for some particles. 
3. **Animation**: Creating an animated representation of the particles. Each particle is displayed as a solid circle, and its historical positions are shown as a trailing line. The animation was saved as a GIF.
4. **Velocity Calculation**: For each particle, calculate the magnitude of its velocity as a function of time. Save these plots to an `img/` folder.
5. **Identifying Acceleration and Deceleration**:
a. Identify decelerations and accelerations produced upon reaching and departing from a target directly (WITHOUT waiting for another person to leave the same target).
b. Select as many of these arrival and departure events as possible (more than 20). For each, record the free walking speed before arrival and after departure from the target (`vdmax`), and the distance to the target at which the arrival maneuver begins (`da`).
6. **Parameter Fitting**:
a. Using the quantities `vdmax` and `da`, fit the parameters 𝜏a (arrival) and 𝜏p (departure) of the chosen model.
b. Perform simulations of a single particle in one dimension, considering only the self-propulsion component (ignoring interactions between particles).
7. **Collision Avoidance Heuristic**
a. Define a heuristic for collision avoidance.
b. Use the value of 𝜏 found in Objective 6 and maximum radii that do not exceed 0.3 m.
c. For SFM, ignore the term for social repulsion.
8. **Inverse Morel Invention**
a. Simulate a single particle interacting with the experimental data trajectories.
b. Report various metrics including the number of experimental particles against which it collided, average speed, and minimum distance to nearest neighbors.


## Tools and Libraries Used
- Python 3.x
- Pandas
- Matplotlib
- NumPy
- OS (for directory manipulation)

## Files
- `src/`: Folder containing the source code.
- `merged_trajectories.txt`: The merged and filtered data file obtained from the preprocessing step.
- `img/`: Folder containing velocity plots for each particle.
- `gif/`: Folder containing the animation GIF.