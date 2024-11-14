# visualize.py
import sys
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from obstacles import SphereObstacle
from astar import a_star_3d
from rrt import rrt
from rrt_star import rrt_star, plot_rrt_attempts
from smooth_path import smooth_path_with_collision_avoidance

def generate_random_spheres_along_line(num_spheres, radius_range, start_pos, goal_pos, spread=5.0, min_distance=5.0, concentration_factor=0.8):
    """
    Generate random spherical obstacles spread across the space, ensuring obstacles are a minimum distance from start and goal positions.

    Args:
        num_spheres (int): Number of spheres.
        radius_range (tuple): Min and max radius for the spheres.
        start_pos (tuple): Starting position in 3D space.
        goal_pos (tuple): Goal position in 3D space.
        spread (float): Amount of spread around the line.
        min_distance (float): Minimum distance from the start or goal positions (includes obstacle radius).
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

        # Ensure obstacle center is not closer than `min_distance + radius` to the start or goal positions
        distance_to_start = np.linalg.norm(center - np.array(start_pos))
        distance_to_goal = np.linalg.norm(center - np.array(goal_pos))

        if (distance_to_start < (min_distance + radius)) or (distance_to_goal < (min_distance + radius)):
            continue  # Skip this obstacle if too close to start or goal

        # Ensure the center is within bounds
        lower_bound = radius  # The minimum allowable position for the center
        upper_bound = np.array(space_dim) - radius  # The maximum allowable position for the center
        if np.any(center < lower_bound) or np.any(center > upper_bound):
            continue  # Skip this obstacle if it goes out of bounds

        # Add the obstacle to the list
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

# Visualize
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

    # Run A* and RRT* algorithms
    import sys
    if len(sys.argv) < 2:
        print("Usage: python visualize.py <algorithm>")
        print("Available algorithms: a_star, rrt, rrt_star, compare")
        sys.exit(1)

    algorithm = sys.argv[1].strip().lower()

    if algorithm == "a_star":
        start_time = time.perf_counter()
        path, visited_positions = a_star_3d(start_pos, goal_pos, obstacles, space_dim)
        end_time = time.perf_counter()
        elapsed_time = (end_time - start_time) * 1000
        if path:
            path_cost = sum(np.linalg.norm(np.array(path[i]) - np.array(path[i + 1])) for i in range(len(path) - 1))
            print(f"A* Path found in {elapsed_time:.2f} ms with cost {path_cost:.2f}")
            smoothed_path = smooth_path_with_collision_avoidance(path, obstacles)
            plot_final_path(space_dim, start_pos, goal_pos, path, smoothed_path, obstacles)
        else:
            print(f"A* failed to find a path in {elapsed_time:.2f} ms.")

    elif algorithm == "rrt_star":
        retries = 5
        start_time = time.perf_counter()
        path, tree = rrt_star(start_pos, goal_pos, obstacles, space_dim, retries=retries)
        end_time = time.perf_counter()
        elapsed_time = (end_time - start_time) * 1000
        if path:
            path_cost = sum(np.linalg.norm(np.array(path[i]) - np.array(path[i + 1])) for i in range(len(path) - 1))
            print(f"RRT* Path found in {elapsed_time:.2f} ms with cost {path_cost:.2f}")
            plot_rrt_attempts(space_dim, start_pos, goal_pos, tree, obstacles)
            smoothed_path = smooth_path_with_collision_avoidance(path, obstacles)
            plot_final_path(space_dim, start_pos, goal_pos, path, smoothed_path, obstacles)
        else:
            print(f"RRT* failed to find a path in {elapsed_time:.2f} ms.")

    elif algorithm == "compare":
        # Compare A* and RRT*
        print("\n=== A* Algorithm ===")
        start_time = time.perf_counter()
        path_a_star, visited_positions = a_star_3d(start_pos, goal_pos, obstacles, space_dim)
        end_time = time.perf_counter()
        a_star_time = (end_time - start_time) * 1000

        if path_a_star:
            smoothed_a_star_path = smooth_path_with_collision_avoidance(path_a_star, obstacles)
            smoothed_a_star_points = np.column_stack(smoothed_a_star_path)  # Combine (x, y, z) into single iterable
            a_star_cost = sum(np.linalg.norm(smoothed_a_star_points[i] - smoothed_a_star_points[i + 1])
                            for i in range(len(smoothed_a_star_points) - 1))
            print(f"A* Smoothed Path found in {a_star_time:.2f} ms with cost {a_star_cost:.2f}")
        else:
            print(f"A* failed to find a path in {a_star_time:.2f} ms.")
            smoothed_a_star_points = None
            a_star_cost = float('inf')

        print("\n=== RRT* Algorithm ===")
        retries = 5
        start_time = time.perf_counter()
        path_rrt_star, tree = rrt_star(start_pos, goal_pos, obstacles, space_dim, retries=retries)
        end_time = time.perf_counter()
        rrt_star_time = (end_time - start_time) * 1000

        if path_rrt_star:
            smoothed_rrt_star_path = smooth_path_with_collision_avoidance(path_rrt_star, obstacles)
            smoothed_rrt_star_points = np.column_stack(smoothed_rrt_star_path)  # Combine (x, y, z) into single iterable
            rrt_star_cost = sum(np.linalg.norm(smoothed_rrt_star_points[i] - smoothed_rrt_star_points[i + 1])
                                for i in range(len(smoothed_rrt_star_points) - 1))
            print(f"RRT* Smoothed Path found in {rrt_star_time:.2f} ms with cost {rrt_star_cost:.2f}")
        else:
            print(f"RRT* failed to find a path in {rrt_star_time:.2f} ms.")
            smoothed_rrt_star_points = None
            rrt_star_cost = float('inf')

        # Plot both smoothed paths on the same graph
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(0, space_dim[0])
        ax.set_ylim(0, space_dim[1])
        ax.set_zlim(0, space_dim[2])

        # Plot obstacles with boundaries
        for obstacle in obstacles:
            plot_obstacle_with_boundary(ax, obstacle, inflation=0.5)

        # Plot start and goal positions
        ax.scatter(*start_pos, color='blue', s=50, label='Start')
        ax.scatter(*goal_pos, color='red', s=50, label='Goal')

        # Plot smoothed A* path
        if smoothed_a_star_points is not None:
            ax.plot(smoothed_a_star_points[:, 0], smoothed_a_star_points[:, 1], smoothed_a_star_points[:, 2],
                    'g--', label=f"A* Smoothed (Cost: {a_star_cost:.2f})")

        # Plot smoothed RRT* path
        if smoothed_rrt_star_points is not None:
            ax.plot(smoothed_rrt_star_points[:, 0], smoothed_rrt_star_points[:, 1], smoothed_rrt_star_points[:, 2],
                    'r-', label=f"RRT* Smoothed (Cost: {rrt_star_cost:.2f})")

        plt.legend()
        plt.title(f"Algorithm Comparison (Smoothed): A* vs RRT*\nA* Time: {a_star_time:.2f} ms, RRT* Time: {rrt_star_time:.2f} ms")
        plt.show()

        # Summary
        print("\n=== Summary ===")
        print(f"A*: Time = {a_star_time:.2f} ms, Smoothed Cost = {a_star_cost:.2f}")
        print(f"RRT*: Time = {rrt_star_time:.2f} ms, Smoothed Cost = {rrt_star_cost:.2f}")

    else:
        print("Invalid algorithm choice. Available options: a_star, rrt, rrt_star, compare.")
        sys.exit(1)