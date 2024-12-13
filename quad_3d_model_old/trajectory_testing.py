import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def trajectory_references(t, point=False, height_i=2, d_height=5, Ts=0.1, innerDyn_length=4, t_final=5):
    """
    Computes trajectory references and derivatives for a system at a given time t.
    
    Parameters:
    - t: float, time in seconds.
    - height_i: float, initial height.
    - d_height: float, height increment over time.
    - Ts: float, sampling time.
    - innerDyn_length: float, dynamic scaling factor.
    - t_final: float, final time for scaling height.

    Returns:
    - np.ndarray: A single 12-dimensional trajectory state vector X^T.
    """
    if point:
        x, y, z, x_dot, y_dot, z_dot= 1, 1, 1, 0, 0, 0
    else:
        # Compute x, y, z trajectories
        x = 0.15 * t + 1 + 2 * np.cos(t / 5)
        y = 0.15 * t - 2 + 2 * np.sin(t / 5)
        z = height_i + d_height/t_final * t + 20 * np.sin(0.3 * t)

        # First derivatives
        x_dot = 0.15 - 0.4 * np.sin(t / 5)
        y_dot = 0.15 + 0.4 * np.cos(t / 5)
        z_dot = (d_height / t_final) + 0.3 * 20 * np.cos(0.3 * t)

        # Second derivatives
        x_dot_dot = -0.06 * np.cos(t / 5)  
        y_dot_dot = -0.06 * np.sin(t / 5)
        z_dot_dot = -1.8 * np.sin(0.3 * t)

    # Create trajectory state vector
    trajectory_state = np.array([
        x, y, z,              # Positions
        0, 0, 0,              # Angles (phi, theta, psi)
        x_dot, y_dot, z_dot,  # Velocities
        0, 0, 0               # Angular rates (phi_dot, theta_dot, psi_dot)
    ])

    return trajectory_state


def plot_ref_trajectory(ax=None):
    """
    Plots the reference trajectory on the given 3D axes.

    Parameters:
    - ax: A Matplotlib 3D axis object to plot on. If None, a new figure and axis are created.

    Returns:
    - ax: The Matplotlib 3D axis object used for the plot.
    """
    # Generate time points
    time_points = np.linspace(0, 10, 101)  # Time from 0 to 10 seconds, 101 samples

    # Initialize trajectory lists
    x_vals, y_vals, z_vals = [], [], []

    # Compute the trajectory for each time point
    for t in time_points:
        state = trajectory_references(t, t_final=10)
        x_vals.append(state[0])  # x
        y_vals.append(state[1])  # y
        z_vals.append(state[2])  # z

    # Create a new figure and 3D axis if ax is None
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    # Plot the reference trajectory
    ax.plot(x_vals, y_vals, z_vals, label="Reference Trajectory", color="blue")

    return ax

def plot_ref_trajectory_with_arrows(num_arrows=10):
    time_points = np.linspace(0, 10, 101)
    x_vals, y_vals, z_vals = [], [], []
    x_dot_vals, y_dot_vals, z_dot_vals = [], [], []

    for t in time_points:
        state = trajectory_references(t, t_final=10)
        x_vals.append(state[0])
        y_vals.append(state[1])
        z_vals.append(state[2])
        x_dot_vals.append(state[3])
        y_dot_vals.append(state[4])
        z_dot_vals.append(state[5])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x_vals, y_vals, z_vals, label="Reference Trajectory", color="blue")

    arrow_indices = np.linspace(0, len(time_points) - 1, num_arrows, dtype=int)
    for idx in arrow_indices:
        ax.quiver(
            x_vals[idx], y_vals[idx], z_vals[idx],  # Starting point of the arrow
            x_dot_vals[idx], y_dot_vals[idx], z_dot_vals[idx],  # Velocity components
            length=1.0, normalize=False, color="red", linewidth=0.5
        )
        
        print(f"x_dot_vals[{idx}]: {x_dot_vals[idx]}, \ny_dot_vals[{idx}]: {y_dot_vals[idx]}, \nz_dot_vals[{idx}]: {z_dot_vals[idx]}")

    ax.set_title("3D Trajectory with Velocity Arrows")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()

if __name__ == "__main__":
    # Plot the trajectory in 3D
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    
    ax = plot_ref_trajectory_with_arrows()
    
    # Labels and title
    # ax.set_title("3D Trajectory Example")
    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")
    # ax.set_zlabel("Z")
    # ax.legend()

    # # Show the plot
    # plt.show()
