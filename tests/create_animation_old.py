# create_animation.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def plot_sphere(ax, center, radius, color='gray', alpha=0.3):
    """Plot a 3D sphere given its center and radius."""
    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    x = radius * np.cos(u) * np.sin(v) + center[0]
    y = radius * np.sin(u) * np.sin(v) + center[1]
    z = radius * np.cos(v) + center[2]
    ax.plot_surface(x, y, z, color=color, alpha=alpha)
    
def plot_obstacle_with_boundary(ax, obstacle, inflation=0.5):
    """
    Plot a solid black sphere for the actual obstacle and a light blue translucent sphere
    for the inflated safety boundary.

    Args:
        ax (Axes3D): The 3D axes to plot on.
        obstacle (SphereObstacle): The obstacle to plot.
        inflation (float): Amount to inflate the obstacle radius for the safety boundary.
    """
    # Plot actual obstacle
    plot_sphere(ax, obstacle.center, obstacle.radius, color='black', alpha=1.0)

    # Plot safety boundary
    inflated_radius = obstacle.radius + inflation
    plot_sphere(ax, obstacle.center, inflated_radius, color='lightblue', alpha=0.3)

def create_animation(space_dim, start_pos, goal_pos, obstacles, visited_positions, smoothed_path):
    """
    Create an animation of the A* pathfinding process.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(0, space_dim[0])
    ax.set_ylim(0, space_dim[1])
    ax.set_zlim(0, space_dim[2])

    # Plot obstacles with boundaries
    for obstacle in obstacles:
        plot_obstacle_with_boundary(ax, obstacle, inflation=0.5)

    # Plot start and goal points
    ax.scatter(*start_pos, color='blue', s=50, label='Start')
    ax.scatter(*goal_pos, color='red', s=50, label='Goal')

    # Initialize visited path plot
    visited_path, = ax.plot([], [], [], 'y.', markersize=1, label='Visited Nodes')

    # Initialize smoothed path plot
    smooth_path, = ax.plot([], [], [], 'r-', linewidth=2, label='Smoothed Path')

    def update(frame):
        """Update function for the animation."""
        if frame < len(visited_positions) and len(visited_positions) > 0:
            # Update visited positions
            x_visited, y_visited, z_visited = zip(*visited_positions[:frame + 1])
            visited_path.set_data(x_visited, y_visited)
            visited_path.set_3d_properties(z_visited)
        elif smoothed_path is not None:
            # Plot the smoothed path once all nodes are visited
            x_smooth, y_smooth, z_smooth = smoothed_path
            smooth_path.set_data(x_smooth, y_smooth)
            smooth_path.set_3d_properties(z_smooth)

    # Total frames: visited_positions + smoothed_path
    total_frames = len(visited_positions) + (len(smoothed_path[0]) if smoothed_path else 0)
    animation = FuncAnimation(fig, update, frames=total_frames, interval=50, repeat=False)

    plt.legend()
    return animation, fig
