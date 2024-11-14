# visualize.py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from obstacles import SphereObstacle
from astar import a_star_3d
from smooth_path import smooth_path_with_collision_avoidance

def generate_random_spheres_along_line(num_spheres, radius_range, start_pos, goal_pos, spread=5.0, min_distance=1.0, concentration_factor=0.2):
    """
    Generate random spherical obstacles spread across the space, biased less towards the line
    between start and goal positions.

    Args:
        num_spheres (int): Number of spheres.
        radius_range (tuple): Min and max radius for the spheres.
        start_pos (tuple): Starting position in 3D space.
        goal_pos (tuple): Goal position in 3D space.
        spread (float): Amount of spread around the line.
        min_distance (float): Minimum distance from the start or goal positions for obstacles.
        concentration_factor (float): Controls concentration along the line (lower spreads the obstacles more).

    Returns:
        list: List of SphereObstacle objects.
    """
    obstacles = []
    line_vector = np.array(goal_pos) - np.array(start_pos)

    while len(obstacles) < num_spheres:
        radius = np.random.uniform(radius_range[0], radius_range[1])
        
        # Generate a random point that is less biased towards the line
        t = np.random.beta(concentration_factor, concentration_factor)
        line_point = np.array(start_pos) + t * line_vector

        # Add a larger random deviation for spread
        deviation = np.random.uniform(-spread, spread, 3)
        center = line_point + deviation

        # Ensure obstacle center is not closer than `min_distance` to the start or goal positions
        if np.linalg.norm(center - np.array(start_pos)) < min_distance + radius or np.linalg.norm(center - np.array(goal_pos)) < min_distance + radius:
            continue  # Skip this obstacle if too close to start or goal

        # Ensure the center is within bounds
        center = np.clip(center, [0, 0, 0], goal_pos)
        
        obstacles.append(SphereObstacle(center=center, radius=radius))
    
    return obstacles

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

def plot_final_path(space_dim, start_pos, goal_pos, path, smoothed_path, obstacles):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(0, space_dim[0])
    ax.set_ylim(0, space_dim[1])
    ax.set_zlim(0, space_dim[2])

    # Plot obstacles with boundaries
    for obstacle in obstacles:
        plot_obstacle_with_boundary(ax, obstacle, inflation=0.5)

    # Plot start, goal, and original path
    ax.scatter(*start_pos, color='blue', s=50, label='Start')
    ax.scatter(*goal_pos, color='red', s=50, label='Goal')
    if path:
        path_x, path_y, path_z = zip(*path)
        ax.plot(path_x, path_y, path_z, 'g--', label='Original Path')

    # Plot the smoothed path if it exists
    if smoothed_path:
        x_smooth, y_smooth, z_smooth = smoothed_path
        ax.plot(x_smooth, y_smooth, z_smooth, 'r-', label='Smoothed Path', linewidth=2)

    plt.legend()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Define 3D space dimensions and parameters
    space_dim = (20, 20, 20)
    num_spheres = 10
    radius_range = (1.5, 2.5)

    # Define start and goal positions
    start_pos = np.array([0, 0, 0])
    goal_pos = np.array([15, 15, 15])

    # Generate obstacles close to the line between start and goal
    obstacles = generate_random_spheres_along_line(num_spheres, radius_range, start_pos, goal_pos, spread=1.5, min_distance=1)

    # Find path using A*
    path, visited_positions = a_star_3d(start_pos, goal_pos, obstacles, space_dim)

    # Check if a path was found
    if path is None:
        print("No path found within the maximum number of iterations.")
    else:
        # Apply smoothing to the path with collision avoidance
        smoothed_path = smooth_path_with_collision_avoidance(path, obstacles)
        # Plot the final path with obstacles
        plot_final_path(space_dim, start_pos, goal_pos, path, smoothed_path, obstacles)
