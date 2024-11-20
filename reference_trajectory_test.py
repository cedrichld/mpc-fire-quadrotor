import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

# Example RRT* path points (replace these with your actual path)
points = np.array([[0, 0, 0], [1, 2, 1], [2, 4, 0], [3, 6, 1], [4, 8, 0]])
v_avg = 1.0  # average velocity (m/s)

# Calculate distances and times
distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
time_intervals = distances / v_avg
timestamps = np.insert(np.cumsum(time_intervals), 0, 0)  # cumulative times

# Interpolation for smooth trajectory
splines = [CubicSpline(timestamps, points[:, dim]) for dim in range(points.shape[1])]

# Reference trajectory functions
def position(t):
    return np.array([spline(t) for spline in splines])

def velocity(t):
    return np.array([spline.derivative(1)(t) for spline in splines])

def acceleration(t):
    return np.array([spline.derivative(2)(t) for spline in splines])

# Plotting
def plot_trajectory():
    # Time vector for plotting
    t_plot = np.linspace(timestamps[0], timestamps[-1], 500)

    # Compute positions, velocities, and accelerations
    pos = np.array([position(t) for t in t_plot])
    vel = np.array([velocity(t) for t in t_plot])
    acc = np.array([acceleration(t) for t in t_plot])

    # Plot position
    plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 1)
    for i, label in enumerate(['x', 'y', 'z']):
        plt.plot(t_plot, pos[:, i], label=f'Position ({label})')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.legend()
    plt.title('Position vs Time')

    # Plot velocity
    plt.subplot(3, 1, 2)
    for i, label in enumerate(['x', 'y', 'z']):
        plt.plot(t_plot, vel[:, i], label=f'Velocity ({label})')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.legend()
    plt.title('Velocity vs Time')

    # Plot acceleration
    plt.subplot(3, 1, 3)
    for i, label in enumerate(['x', 'y', 'z']):
        plt.plot(t_plot, acc[:, i], label=f'Acceleration ({label})')
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (m/s²)')
    plt.legend()
    plt.title('Acceleration vs Time')

    plt.tight_layout()
    plt.show()

# Call the plot function
plot_trajectory()
