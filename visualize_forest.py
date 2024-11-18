# visualize_forest.py
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import splprep, splev
from obstacles import CylinderObstacle, is_point_in_collision
from astar import a_star_2d, a_star_3d
from rrt_star import rrt_star, plot_rrt_attempts
from smooth_path import smooth_path_with_collision_avoidance

def generate_random_forest_with_grid(
    grid_size, radius_range, height_range, space_dim, zone_center, 
    zone_radius, start_pos, min_start_distance
):
    """
    Generate a forest of cylindrical obstacles (Trees) using a grid-based approach.

    Args:
        grid_size (int): Number of grid cells along one axis (total grid cells = grid_size^2).
        radius_range (tuple): Min and max radius for tree cylinders.
        height_range (tuple): Min and max height for tree cylinders.
        space_dim (tuple): Dimensions of the 3D space (x, y, z).
        zone_center (tuple): Center of the fire zone (x, y).
        zone_radius (float): Approximate radius for the fire zone boundary.
        start_pos (tuple): Start position (x, y, z).
        min_start_distance (float): Minimum allowed distance between trees and the start position.

    Returns:
        tuple: (list of CylinderObstacle objects, list of CylinderObstacle objects in fire zone)
    """
    obstacles = []
    fire_zone_trees = []

    # Calculate the size of each grid cell
    grid_cell_size_x = space_dim[0] / grid_size
    grid_cell_size_y = space_dim[1] / grid_size

    # Iterate over grid cells
    for i in range(grid_size):
        for j in range(grid_size):
            # Compute the center of the grid cell
            cell_min_x = i * grid_cell_size_x
            cell_max_x = (i + 1) * grid_cell_size_x
            cell_min_y = j * grid_cell_size_y
            cell_max_y = (j + 1) * grid_cell_size_y

            # Randomly place a tree within the grid cell
            radius = np.random.uniform(*radius_range)
            height = np.random.uniform(*height_range)
            x = np.random.uniform(cell_min_x + radius, cell_max_x - radius)
            y = np.random.uniform(cell_min_y + radius, cell_max_y - radius)

            center = np.array([x, y, 0])

            # Check if the tree is too close to the start position
            if np.linalg.norm(center[:2] - np.array(start_pos[:2])) < (radius + min_start_distance):
                continue

            # Check if the tree is in the fire zone
            distance_to_fire_center = np.linalg.norm(np.array(zone_center) - center[:2])
            if distance_to_fire_center < zone_radius:
                fire_zone_trees.append(CylinderObstacle(center=center, height=height, radius=radius))
            else:
                obstacles.append(CylinderObstacle(center=center, height=height, radius=radius))

    return obstacles, fire_zone_trees

def generate_fire_zone(center, size):
    """
    Generate a fire zone represented as a closed 2D spline approximating an ellipse.

    Args:
        center (tuple): Approximate center of the fire zone.
        size (float): Approximate size of the fire zone.

    Returns:
        tuple: (x, y) coordinates of the fire zone.
    """
    angle = np.linspace(0, 2 * np.pi, 10, endpoint=False)
    radii = size * (0.8 + 0.4 * np.random.rand(len(angle)))  # Randomized radii for variation
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

def adjust_goal_position_smoothly(goal_pos, obstacles, inflation, step_size=0.5, max_attempts=50):
    """
    Adjust the goal position to ensure it is not in collision.

    Args:
        goal_pos (tuple): Initial goal position (x, y).
        obstacles (list): List of CylinderObstacle objects.
        inflation (float): Amount to inflate the obstacle radii for safety.
        step (float): Step size for searching nearby valid positions.
        max_attempts (int): Maximum number of attempts to adjust the position.

    Returns:
        tuple: Adjusted goal position (x, y).
    """
    for _ in range(max_attempts):
        if not is_point_in_collision(goal_pos, obstacles, inflation):
            return goal_pos  # Valid position found
        # Move goal slightly in random direction
        direction = np.random.uniform(-1, 1, size=2)
        goal_pos = np.array(goal_pos) + step_size * direction / np.linalg.norm(direction)
    raise RuntimeError("Failed to adjust goal position within constraints.")

def visualize_forest(space_dim, obstacles, fire_zone, start_pos, goal_pos, fire_zone_trees=None):
    """
    Visualize the forest environment with cylindrical obstacles, fire zone, and start/goal points.

    Args:
        space_dim (tuple): Dimensions of the 3D space (x, y, z).
        obstacles (list): List of CylinderObstacle objects outside the fire zone.
        fire_zone (tuple): Coordinates of the fire zone (x, y).
        start_pos (tuple): Start position of the drone.
        goal_pos (tuple): Goal position of the drone.
        fire_zone_trees (list): Trees inside the fire zone, displayed in orange. (Optional)
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(0, space_dim[0])
    ax.set_ylim(0, space_dim[1])
    ax.set_zlim(0, space_dim[2])

    # Plot trees outside fire zone
    for obstacle in obstacles:
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, 1, 10)
        x = obstacle.radius * np.outer(np.cos(u), np.ones(len(v))) + obstacle.center[0]
        y = obstacle.radius * np.outer(np.sin(u), np.ones(len(v))) + obstacle.center[1]
        z = obstacle.height * np.outer(np.ones(len(u)), v) + obstacle.center[2]
        ax.plot_surface(x, y, z, color='green', alpha=0.5)

    # Plot trees inside fire zone (if provided)
    if fire_zone_trees:
        for obstacle in fire_zone_trees:
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, 1, 10)
            x = obstacle.radius * np.outer(np.cos(u), np.ones(len(v))) + obstacle.center[0]
            y = obstacle.radius * np.outer(np.sin(u), np.ones(len(v))) + obstacle.center[1]
            z = obstacle.height * np.outer(np.ones(len(u)), v) + obstacle.center[2]
            ax.plot_surface(x, y, z, color='orange', alpha=0.7)

    # Plot fire zone as filled red area
    fire_x, fire_y = fire_zone
    fire_z = np.zeros_like(fire_x)
    ax.plot(fire_x, fire_y, fire_z, color='red', linewidth=2, label='Fire Zone')

    # Plot start and goal positions
    ax.scatter(*start_pos, color='blue', s=100, label='Start')
    ax.scatter(*goal_pos, color='yellow', s=100, label='Goal')

    plt.legend()
    plt.show()

def visualize_forest_2d(space_dim, obstacles, fire_zone, start_pos, goal_pos, fire_zone_trees=None):
    """
    Visualize the forest environment in 2D with cylindrical obstacles, fire zone, and start/goal points.

    Args:
        space_dim (tuple): Dimensions of the 2D space (x, y).
        obstacles (list): List of CylinderObstacle objects outside the fire zone.
        fire_zone (tuple): Coordinates of the fire zone (x, y).
        start_pos (tuple): Start position of the drone.
        goal_pos (tuple): Goal position of the drone.
        fire_zone_trees (list): Trees inside the fire zone, displayed in orange. (Optional)
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, space_dim[0])
    ax.set_ylim(0, space_dim[1])

    # Plot trees outside fire zone in green
    for obstacle in obstacles:
        circle = plt.Circle(
            obstacle.center[:2], obstacle.radius, color='green', alpha=0.5
        )
        ax.add_patch(circle)

    # Plot trees inside fire zone in orange
    if fire_zone_trees:
        for obstacle in fire_zone_trees:
            circle = plt.Circle(
                obstacle.center[:2], obstacle.radius, color='orange', alpha=0.7
            )
            ax.add_patch(circle)

    # Plot fire zone boundary
    fire_x, fire_y = fire_zone
    ax.plot(fire_x, fire_y, color='red', linewidth=2, label='Fire Zone')

    # Plot start and goal positions
    ax.scatter(start_pos[0], start_pos[1], color='blue', s=100, label='Start')
    ax.scatter(goal_pos[0], goal_pos[1], color='yellow', s=100, label='Goal')
    
    return fig, ax
    
# Implementation
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Visualize forest environment for path planning.")
    parser.add_argument("mode", choices=["2d", "3d", "a_star_2d", "rrt_star_2d"], 
        help="Visualization mode: '2d' for 2D visualization, '3d' for 3D visualization, 'a_star' for pathfinding.")
    args = parser.parse_args()

    # Define 3D space dimensions and parameters
    space_dim = (50, 50, 20)
    grid_size = 25  # Define a grid
    radius_range = (0.1, 0.25)
    height_range = (5.0, 15.0)
    zone_center = (25, 25)
    zone_radius = 10
    
    # Generate fire zone
    fire_zone = generate_fire_zone(center=zone_center, size=zone_radius)
    
    # Define start and goal positions
    start_pos = (5, 5, 0)
    closest_fire_point = closest_point_on_spline(fire_zone, start_pos[:2])
    goal_pos = (*closest_fire_point, 0)

    min_start_distance = 1.0  # Minimum distance from start position to any tree

    # Generate forest using grid-based approach
    trees_outside, trees_inside = generate_random_forest_with_grid(
        grid_size, radius_range, height_range, space_dim, zone_center, zone_radius, start_pos, min_start_distance
    )

    # Handle different modes
    if args.mode == "3d":
        visualize_forest(space_dim, trees_outside, fire_zone, start_pos, goal_pos, trees_inside)
    elif args.mode == "2d":
        visualize_forest_2d(space_dim[:2], trees_outside, fire_zone, start_pos, goal_pos, trees_inside)
    
    ## A*
    elif args.mode == "a_star_2d":
        # Combine trees for obstacle input
        obstacles = trees_outside + trees_inside
        
        # Run A* Pathfinding
        print("\n=== A* Pathfinding ===")

        # Adjust goal position if it is in collision
        adjusted_goal_pos = adjust_goal_position_smoothly(
            goal_pos[:2], obstacles, inflation=0.05, step_size=0.5, max_attempts=50
        )
        
        if adjusted_goal_pos is None:
            print("No valid goal position could be determined.")
            exit()
        goal_pos = (*adjusted_goal_pos, 0)
        
        visualize_forest_2d(space_dim[:2], trees_outside, fire_zone, start_pos, goal_pos, trees_inside)
        
        start_time = time.perf_counter()
        path, visited_positions = a_star_2d(start_pos[:2], goal_pos[:2], obstacles, space_dim[:2], eps=0.5)
        end_time = time.perf_counter()
        elapsed_time = (end_time - start_time) * 1000

        if path:
            path_cost = sum(np.linalg.norm(np.array(path[i]) - np.array(path[i + 1])) for i in range(len(path) - 1))
            print(f"A* Path found in {elapsed_time:.2f} ms with cost {path_cost:.2f}")

            # Smooth the path
            smoothed_path = smooth_path_with_collision_avoidance(path, obstacles)
            smoothed_path_x, smoothed_path_y = smoothed_path[:2]

            # Visualize in 2D
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.set_xlim(0, space_dim[0])
            ax.set_ylim(0, space_dim[1])

            
            path_x, path_y = zip(*path)
            ax.plot(path_x, path_y, color='purple', linewidth=2, label='A* Path')

            # Smoothed path
            ax.plot(smoothed_path_x, smoothed_path_y, color='cyan', linestyle='--', linewidth=2, label='Smoothed Path')
        else:
            print(f"A* failed to find a path in {elapsed_time:.2f} ms.")
    
    ## RRT*
    elif args.mode == "rrt_star_2d":
        # Combine trees for obstacle input
        obstacles = trees_outside + trees_inside

        # Run RRT* Pathfinding
        print("\n=== RRT* Pathfinding ===")

        # Adjust goal position if it is in collision
        adjusted_goal_pos = adjust_goal_position_smoothly(
            goal_pos[:2], obstacles, inflation=0.05, step_size=0.5, max_attempts=50
        )
        if adjusted_goal_pos is None:
            print("No valid goal position could be determined.")
            exit()
        goal_pos = (*adjusted_goal_pos, 0)

        fig, ax = visualize_forest_2d(space_dim[:2], trees_outside, fire_zone, start_pos, goal_pos, trees_inside)
        
        start_time = time.perf_counter()
        path, tree = rrt_star(
            start_pos[:2], goal_pos[:2], obstacles, space_dim, dim=2
        )

        end_time = time.perf_counter()
        elapsed_time = (end_time - start_time) * 1000

        if path:
            path_cost = sum(np.linalg.norm(np.array(path[i]) - np.array(path[i + 1])) for i in range(len(path) - 1))
            print(f"RRT* Path found in {elapsed_time:.2f} ms with cost {path_cost:.2f}")

            # Smooth the path
            smoothed_path = smooth_path_with_collision_avoidance(path, obstacles)
            smoothed_path_x, smoothed_path_y = smoothed_path[:2]

            # Visualize in 2D
            path_x, path_y = zip(*path)
            ax.plot(path_x, path_y, color='purple', linewidth=2, label='RRT* Path')

            # Smoothed path
            ax.plot(smoothed_path_x, smoothed_path_y, color='cyan', linestyle='--', linewidth=2, label='Smoothed Path')
        else:
            print(f"RRT* failed to find a path in {elapsed_time:.2f} ms.")
        
        # Plot RRT attempts
        plot_rrt_attempts(ax, tree, dim=2)
            
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
