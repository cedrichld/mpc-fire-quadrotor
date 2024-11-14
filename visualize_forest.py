import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import splprep, splev
from obstacles import CylinderObstacle

def generate_random_forest(num_trees, radius_range, height_range, space_dim, fire_zone, fire_zone_radius=12.0, min_distance=5.0):
    """
    Generate a forest of cylindrical obstacles with proportional height and diameter, ensuring no overlap with the fire zone.

    Args:
        num_trees (int): Number of trees to generate.
        radius_range (tuple): Min and max radius for tree cylinders.
        height_range (tuple): Min and max height for tree cylinders.
        space_dim (tuple): Dimensions of the 3D space (x, y, z).
        fire_zone (tuple): Coordinates of the fire zone (x, y).
        fire_zone_radius (float): Radius of the circular buffer around the fire zone.
        min_distance (float): Minimum distance between trees and start/goal points.

    Returns:
        list: List of CylinderObstacle objects.
    """
    obstacles = []
    fire_x, fire_y = fire_zone
    fire_center = np.mean(np.column_stack((fire_x, fire_y)), axis=0)

    while len(obstacles) < num_trees:
        # Generate height and compute proportional radius
        height = np.random.uniform(*height_range)
        radius = np.interp(height, height_range, radius_range)  # Scale radius based on height

        # Generate random base center
        base_center = np.random.uniform([0, 0], [space_dim[0], space_dim[1]])

        # Ensure no overlap with the fire zone buffer
        distance_to_fire_center = np.linalg.norm(base_center - fire_center)
        if distance_to_fire_center < fire_zone_radius + radius:
            continue

        # Ensure no overlap with existing obstacles
        if any(
            np.linalg.norm(base_center - other.base_center[:2]) < (radius + other.radius + min_distance)
            for other in obstacles
        ):
            continue

        # Add obstacle to the list
        obstacles.append(CylinderObstacle(base_center=np.append(base_center, 0), height=height, radius=radius))

    return obstacles

def generate_fire_zone(center, size):
    """
    Generate a fire zone represented as a closed 2D spline approximating an ellipse.

    Args:
        center (tuple): Approximate center of the fire zone.
        size (float): Approximate size of the fire zone.

    Returns:
        tuple: (x, y) coordinates of the fire zone.
    """
    np.random.seed(42)
    angle = np.linspace(0, 2 * np.pi, 10, endpoint=False)
    radii = size * (0.8 + 0.4 * np.random.rand(len(angle)))
    control_points = np.column_stack((radii * np.cos(angle), radii * np.sin(angle))) + center[:2]
    tck, _ = splprep(control_points.T, s=0.5, per=True)
    u = np.linspace(0, 1, 100)
    return splev(u, tck)

def closest_point_on_spline(spline, point):
    """
    Find the closest point on a spline to a given point.

    Args:
        spline (tuple): The spline coordinates (x, y).
        point (tuple): The target point (x, y).

    Returns:
        tuple: Closest point on the spline (x, y).
    """
    x_spline, y_spline = spline
    distances = np.sqrt((np.array(x_spline) - point[0])**2 + (np.array(y_spline) - point[1])**2)
    closest_index = np.argmin(distances)
    return x_spline[closest_index], y_spline[closest_index]

def visualize_forest(space_dim, obstacles, fire_zone, start_pos, goal_pos):
    """
    Visualize the forest environment with cylindrical obstacles, fire zone, and start/goal points.

    Args:
        space_dim (tuple): Dimensions of the 3D space (x, y, z).
        obstacles (list): List of CylinderObstacle objects.
        fire_zone (tuple): Coordinates of the fire zone (x, y).
        start_pos (tuple): Start position of the drone.
        goal_pos (tuple): Goal position of the drone.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(0, space_dim[0])
    ax.set_ylim(0, space_dim[1])
    ax.set_zlim(0, space_dim[2])

    # Plot cylindrical obstacles
    for obstacle in obstacles:
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, 1, 10)
        x = obstacle.radius * np.outer(np.cos(u), np.ones(len(v))) + obstacle.base_center[0]
        y = obstacle.radius * np.outer(np.sin(u), np.ones(len(v))) + obstacle.base_center[1]
        z = obstacle.height * np.outer(np.ones(len(u)), v) + obstacle.base_center[2]
        ax.plot_surface(x, y, z, color='green', alpha=0.5)

    # Plot fire zone
    fire_x, fire_y = fire_zone
    fire_z = np.zeros_like(fire_x)
    ax.plot(fire_x, fire_y, fire_z, color='red', linewidth=2, label='Fire Zone')

    # Plot start and goal positions
    ax.scatter(*start_pos, color='blue', s=100, label='Start')
    ax.scatter(*goal_pos, color='yellow', s=100, label='Goal')

    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Define 3D space dimensions and parameters
    space_dim = (50, 50, 20)
    num_trees = 30
    radius_range = (0.5, 2.0)
    height_range = (5.0, 15.0)

    # Generate fire zone
    fire_zone = generate_fire_zone(center=(25, 25), size=10)

    # Generate forest
    obstacles = generate_random_forest(num_trees, radius_range, height_range, space_dim, fire_zone)

    # Define start and goal positions
    start_pos = (5, 5, 0)
    closest_fire_point = closest_point_on_spline(fire_zone, start_pos[:2])
    goal_pos = (*closest_fire_point, 0)

    # Visualize forest
    visualize_forest(space_dim, obstacles, fire_zone, start_pos, goal_pos)
